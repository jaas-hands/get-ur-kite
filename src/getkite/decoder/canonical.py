"""CF: run a decoder with every robot expressed in its canonical palm frame (docs/cf_canonical_palm_frame_proposal.md).

CanonicalFrame wraps a model with the decoder call signature model(bank, r, qc0, R0, t0, z). For any bank it builds (once,
cached by robot names and device) the canonical bank (robots.canonical_bank: the same robots, base frames re-expressed by
Q_r, world geometry unchanged). It calls the inner model on that bank with R0 -> R0 Q_r and maps the output back
(R -> R' Q_r^T). res_root is re-expressed in the caller's frame via target_residuals, so evaluate()'s residual diagnostic
compares like with like. Q_r comes from the robot description and template only (a deployment-time input).
"""
from __future__ import annotations

import torch
from torch import nn

from .get_scratch import target_residuals
from .robots import canonical_bank


class CanonicalFrame(nn.Module):
    """origin=True (CF-O, docs/canonical_origin_proposal.md): the base origin is also canonical (o_r, native frame); start
    pose (R0, t0) -> (R0 Q_r, t0 + R0 o_r), output (R', t') -> (R' Q_r^T, t' - R o_r). Cache entries: (bank_c, Q) or
    (bank_c, Q, O)."""

    def __init__(self, inner: nn.Module, origin: bool = False):
        super().__init__()
        self.inner = inner
        self.origin = bool(origin)
        self.c = inner.c
        self._cache = {}

    @property
    def contacts(self):
        return self.inner.contacts

    def predict_contacts(self, z):
        return self.inner.predict_contacts(z)

    def canon(self, bank):
        key = (id(bank), tuple(bank.names), str(bank.lower.device))  # id: a relabelled bank has the same names
        if key not in self._cache:
            bc, Q, info = canonical_bank(bank, origin=self.origin)
            O = torch.as_tensor([i["o"] for i in info], dtype=Q.dtype, device=Q.device) if self.origin else None
            self._cache[key] = (bc, Q, O)
        return self._cache[key]

    def forward(self, bank, r, qc0, R0, t0, z):
        c = self.canon(bank)
        bc, Q, O = c if len(c) == 3 else (*c, None)
        Qr = Q[r]
        if O is not None:  # CF-O: canonical origin
            Or = O[r]
            out = self.inner(bc, r, qc0, R0 @ Qr, t0 + (R0 @ Or[..., None]).squeeze(-1), z)
            out["R"] = out["R"] @ Qr.transpose(1, 2)
            out["t"] = out["t"] - (out["R"] @ Or[..., None]).squeeze(-1)
        else:
            out = self.inner(bc, r, qc0, R0 @ Qr, t0, z)
            out["R"] = out["R"] @ Qr.transpose(1, 2)
        out["res_root"] = target_residuals(bank, r, qc0, R0, t0, out["qc"], out["R"], out["t"], self.c)[0]
        return out
