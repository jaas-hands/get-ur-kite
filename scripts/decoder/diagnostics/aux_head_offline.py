"""Offline re-fit of the direct decoder's aux head on its FROZEN input features (2026-11-17, diagnostics only).

Same architecture as GETScratch.aux_head and the same SetMatcher loss / wrist-frame labels, trained on frozen
features of the C3 training pool (fixed start), scored on fresh goals. Separates "head + set loss cannot fit these
features" from "joint training dynamics prevented it".
"""
import importlib.util, json, sys
from pathlib import Path
import torch, torch.nn as nn
spec = importlib.util.spec_from_file_location("ap", Path(__file__).with_name("aux_probe.py")); ap = importlib.util.module_from_spec(spec); spec.loader.exec_module(ap)
from getkite.decoder.get_scratch import GETScratch, ScratchConfig
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.spatial_bridge import SetMatcher, contact_set_label

dev = torch.device("cuda")
cfg = ap.load_config(f"{ap.REPO}/configs/decoder/scratch/c3_psi50k_10x_wrist.json", [])
bank = ap.build_bank(cfg, dev); psi = ap.FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval(); noise = ap.start_noise(cfg)
train = ap.load_pool(cfg["train"], bank).to(dev)
train = train.index(torch.randperm(len(train), generator=torch.Generator().manual_seed(3))[:30000].to(dev))
val = ap.load_pool(cfg["val"], bank).to(dev)
def sg_of(rec, seed):
    s = fixed_start(bank, rec, seed, **noise); return type(s)(**{k: getattr(s, k).to(dev) for k in s.__dataclass_fields__})
sgt, sgv = sg_of(train, cfg.get("train_start_seed", 0) + 17), sg_of(val, cfg["val_start_seed"])
ck = torch.load(f"{ap.RUNS}/scratch_c3_aux50k_10x_wrist/best.pt", map_location=dev, weights_only=False)
m = GETScratch(ScratchConfig.from_dict(ck["model_config"])).to(dev); m.load_state_dict(ck["model"]); m.amp = False; m.eval()
Ft, Fv = ap.features(m, psi, bank, train, sgt), ap.features(m, psi, bank, val, sgv)
Lt, Lv = contact_set_label(train.c_pos, train.c_normal, train.c_valid, sgt.R1, sgt.t1), contact_set_label(val.c_pos, val.c_normal, val.c_valid, sgv.R1, sgv.t1)
M = SetMatcher().to(dev)
out = {}
for feat, lr in (("aux_in", 1e-3), ("aux_in", 3e-4), ("final_all", 1e-3)):
    torch.manual_seed(0)
    X = Ft[feat]; d = X.shape[1]
    head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 42)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    def run(Xb):
        a = head(Xb).view(len(Xb), 6, 7)
        return a[..., :3] * 0.1, torch.nn.functional.normalize(a[..., 3:6], dim=-1), a[..., 6]
    for step in range(6000):
        j = torch.randint(0, len(X), (256,), device=dev)
        L, _ = M.set_loss(*run(X[j]), Lt[0][j], Lt[1][j], Lt[2][j])
        opt.zero_grad(); L["total"].backward(); opt.step()
    res = {}
    with torch.no_grad():
        for nm, Xe, Le, n in (("train", X[:5000], [l[:5000] for l in Lt], 5000), ("val", Fv[feat], Lv, len(Fv[feat]))):
            p = run(Xe); L, _ = M.set_loss(*p, *Le)
            res[nm] = {"count_acc": round(float(((p[2].sigmoid() > 0.5).sum(1) == Le[2].sum(1)).float().mean()), 3),
                       "pos_rms_mm": round(float(L["pos"]) ** 0.5 * 10, 1), "exist_bce": round(float(L["exist"]), 3)}
    out[f"{feat}@lr{lr}"] = res; print(feat, lr, res, flush=True)
out["own_joint_trained_head_count_acc_val"] = 0.247
Path(f"{ap.RUNS}/aux_probe/offline_head.json").write_text(json.dumps(out, indent=1))
