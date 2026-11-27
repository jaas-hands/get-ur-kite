"""Read-only: which in-training (log) metric at the selected checkpoint predicts real new-family error (Wuji + Shadow, and the
7 external hands where available)? Every trained J1 model with a benchmark entry counts once (dedup by checkpoint)."""
import glob, json, os, re, numpy as np
from scipy.stats import spearmanr
R = os.environ["GETKITE_RUNS"]
models = {}
for d in sorted(glob.glob(f"{R}/transfer_benchmark_v1*/")):
    if re.search(r"leapval105|leap360", d) or not os.path.exists(d + "metrics.json") or not os.path.exists(d + "manifest.json"):
        continue
    M = json.load(open(d + "metrics.json")); man = json.load(open(d + "manifest.json"))
    sysm = man.get("benchmark", {}).get("systems", {})
    for name, spec in sysm.items():
        ck = spec.get("ckpt") if isinstance(spec, dict) else None
        if not ck or not ck.endswith("best.pt"):
            continue
        run = os.path.dirname(ck.replace("${GETKITE_RUNS}", R))
        run = os.path.realpath(run)
        w = M.get("wuji", {}).get("wuji", {}).get(name); s = M.get("shadow", {}).get("shadow", {}).get(name)
        if not isinstance(w, dict):
            continue
        e = models.setdefault(run, {"names": set()})
        e["names"].add(f"{os.path.basename(d.rstrip('/'))}:{name}")
        for k, v in (("wuji_corr", w["corr_mm"]), ("wuji_wrist", w["wrist_mm"])) + ((("shadow_corr", s["corr_mm"]), ("shadow_wrist", s["wrist_mm"])) if isinstance(s, dict) else ()):
            e.setdefault(k, []).append(v)
# consistency of repeated entries
incons = [(r, k, v) for r, e in models.items() for k, v in e.items() if k != "names" and max(v) - min(v) > 1e-3]
print("models with benchmark entries:", len(models), "| inconsistent repeats:", len(incons), incons[:3])
# external hands
ext = {}
for f in glob.glob(f"{R}/external_hands_*/external_hands.json"):
    J = json.load(open(f))
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_hands_eval.py")).read()
    for arm in J["arms"]:
        m = re.search(rf'"{arm}": "([^"]+)"', src)
        base = m.group(1) if m else ({f"cf3f{a}": f"cf3f_{a}_hf" for a in ("leap","long","teach","big","rec")}.get(arm))
        if not base: continue
        for s in range(3):
            run = os.path.realpath(f"{R}/{base}{'' if s == 0 else f'_s{s}'}")
            hands = [h for h in J["pools"] if h not in ("wuji", "shadow")]
            v = [J["pools"][h][arm]["seeds"]["corr_mm"][s] for h in hands if arm in J["pools"][h]]
            if len(v) == 7: ext.setdefault(run, []).append(float(np.mean(v)))
SIG = ["all/model/contact_mm", "robot_macro/model/contact_mm", "allegro/model/contact_mm", "barrett/model/contact_mm",
       "leap/model/contact_mm", "prim/model/contact_mm", "leap_heldout:all/model/contact_mm", "prim_heldout:all/model/contact_mm",
       "new_hands_fresh:all/model/contact_mm", "train_goals_new:all/model/contact_mm"]
rows = []
for run, e in models.items():
    try:
        best = json.load(open(f"{run}/DONE"))["best"]["step"]
    except Exception:
        continue
    ev = None; tl = []
    for line in open(f"{run}/log.jsonl"):
        r = json.loads(line)
        if "eval" in r and r["step"] == best: ev = r["eval"]
        if "train/loss" in r and best - 5000 < r["step"] <= best: tl.append(r["train/loss"])
    if ev is None or "shadow_corr" not in e: continue
    row = {"run": os.path.basename(run), "target_ws": (np.mean(e["wuji_corr"]) + np.mean(e["shadow_corr"])) / 2,
           "target_ws_wrist": (np.mean(e["wuji_wrist"]) + np.mean(e["shadow_wrist"])) / 2,
           "ext7": np.mean(ext[run]) if run in ext else None, "train_loss": np.mean(tl) if tl else None}
    for k in SIG: row[k] = ev.get(k)
    rows.append(row)
print("models usable (log at selected ckpt + Wuji + Shadow):", len(rows), "| with 7 external hands:", sum(r["ext7"] is not None for r in rows))
print("runs:", sorted(r["run"] for r in rows))
for tgt in ("target_ws", "target_ws_wrist", "ext7"):
    print(f"\nSpearman vs {tgt}:")
    for k in SIG + ["train_loss"]:
        pts = [(r[k], r[tgt]) for r in rows if r.get(k) is not None and r.get(tgt) is not None]
        if len(pts) < 8: continue
        x, y = np.array(pts).T; s = spearmanr(x, y)
        print(f"  {k:42s} n={len(pts):3d} rho={s.correlation:+.2f} p={s.pvalue:.3f}")
json.dump(rows, open(os.path.join(R, "proxy_signal_rows.json"), "w"), indent=1, default=float)
