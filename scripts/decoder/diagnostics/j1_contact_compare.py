"""Like-for-like contact-stage comparison: J1's internal contact predictor vs the standalone world-frame translator
(2026-11-18). Both evaluated with train_joint.contact_eval (same matching, same pools, frozen psi). No training.

    python scripts/decoder/diagnostics/j1_contact_compare.py --out $GETKITE_RUNS/j1_joint50k_hf/contact_compare.json
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn

from getkite.decoder import paths
from getkite.decoder.joint_contact_get import JointContactGET
from getkite.decoder.spatial_bridge import ContactSetBridge, SetBridgeConfig
from getkite.decoder.get_scratch import ScratchConfig
from getkite.decoder.train_joint import contact_eval
from getkite.decoder.train_scratch import FrozenPsi, build_bank, load_config, load_pool

RUNS = paths.get("GETKITE_RUNS")


class BridgeShim(nn.Module):
    def __init__(self, bridge):
        super().__init__(); self.contacts = bridge

    def predict_contacts(self, z):
        B = len(z)
        eye = torch.eye(3, device=z.device, dtype=z.dtype).expand(B, 3, 3)
        return self.contacts(z, eye, torch.zeros(B, 3, device=z.device, dtype=z.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--j1", default=f"{RUNS}/j1_joint50k_hf/best.pt")
    ap.add_argument("--bridge", default=f"{RUNS}/set_bridge_zworld_c3_20k_b/best.pt")
    ap.add_argument("--out", default=f"{RUNS}/j1_joint50k_hf/contact_compare.json")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.j1, map_location=dev, weights_only=False)
    assert ck.get("kind") == "joint_contact_get"
    cfg = load_config(f"{RUNS}/j1_joint50k_hf/config.json", []) if "config" not in ck else ck["config"]
    bank = build_bank(cfg, dev)
    j1 = JointContactGET(ScratchConfig.from_dict(ck["model_config"]), SetBridgeConfig(**ck["set_config"])).to(dev)
    j1.load_state_dict(ck["model"]); j1.eval()
    bs_ = torch.load(a.bridge, map_location=dev, weights_only=False)
    br = ContactSetBridge(SetBridgeConfig(**bs_["config"])).to(dev); br.load_state_dict(bs_["model"])
    shim = BridgeShim(br).to(dev).eval()
    psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    pools = {"val": cfg["val"], **cfg.get("extra_eval", {})}
    R = {"_meta": {"j1": a.j1, "j1_step": ck.get("step"), "bridge": a.bridge}}
    with torch.no_grad():
        for name, spec in pools.items():
            rec = load_pool(spec, bank).to(dev)
            fams = sorted(set(bank.family[int(i)] for i in rec.r.unique()))
            R[name] = {}
            for tag, m in (("j1", j1), ("standalone", shim)):
                R[name][tag] = {"all": contact_eval(m, psi, rec)}
                for f in fams:
                    idx = torch.tensor([i for i, r in enumerate(rec.r.tolist()) if bank.family[r] == f], device=dev)
                    R[name][tag][f] = contact_eval(m, psi, rec.index(idx))
                m.eval()
            print(name, json.dumps({t: {k: round(v, 3) for k, v in R[name][t]["all"].items()} for t in R[name]}), flush=True)
    open(a.out, "w").write(json.dumps(R, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
