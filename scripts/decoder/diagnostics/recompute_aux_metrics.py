"""Recompute the direct decoder's aux-head metrics per example (bug fix 2026-11-17) on each checkpoint's own
evaluation pools and fixed starts (val + extra_eval, as train_scratch evaluates them). Diagnostics only."""
import json, sys
from pathlib import Path
import numpy as np, torch
from getkite.decoder import paths
from getkite.decoder.get_scratch import GETScratch, ScratchConfig
from getkite.decoder.multifamily import fixed_start
from getkite.decoder.train_scratch import FrozenPsi, build_bank, evaluate, load_config, load_pool, start_noise
RUNS = paths.get("GETKITE_RUNS")
dev = torch.device("cuda")
out = {}
for run in ("scratch_aux_wrist_c2_20k", "scratch_c3_aux50k_10x_wrist"):
    cfg = load_config(f"{RUNS}/{run}/config.json", [])
    cfg.setdefault("eval_batch", 512); cfg.setdefault("success_pos_mm", 10.0); cfg.setdefault("success_normal_deg", 15.0)
    bank = build_bank(cfg, dev); psi = FrozenPsi(cfg["kite_ckpt"], cfg["kite_root"]).to(dev).eval()
    families = sorted(set(bank.family)); noise = start_noise(cfg)
    train = load_pool(cfg["train"], bank).to(dev)
    pools = {"val": load_pool(cfg["val"], bank).to(dev) if cfg.get("val") else train}
    pools |= {k: load_pool(v, bank).to(dev) for k, v in cfg.get("extra_eval", {}).items()}
    for ckn in ("best", "last"):
        ck = torch.load(f"{RUNS}/{run}/{ckn}.pt", map_location=dev, weights_only=False)
        m = GETScratch(ScratchConfig.from_dict(ck["model_config"])).to(dev); m.load_state_dict(ck["model"]); m.amp = False
        key = f"{run}/{ckn}@{ck['step']}"; out[key] = {}
        for pn, rec in pools.items():
            sg = fixed_start(bank, rec, cfg["val_start_seed"], **noise)
            sg = type(sg)(**{k: getattr(sg, k).to(dev) for k in sg.__dataclass_fields__})
            with torch.no_grad():
                res = evaluate(m, psi, bank, rec, sg, cfg, families)
            out[key][pn] = {f: {k: round(res[f"{f}/model/{k}"], 3) for k in ("aux_pos_mm", "aux_normal_deg", "aux_count_acc", "contact_mm")}
                            for f in families if f"{f}/model/aux_pos_mm" in res}
            print(key, pn, json.dumps(out[key][pn]), flush=True)
Path(f"{RUNS}/aux_probe/aux_metrics_recomputed.json").write_text(json.dumps(out, indent=1))
