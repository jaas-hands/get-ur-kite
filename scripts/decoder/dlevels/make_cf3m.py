"""CF-3m: teacher-memorisation v2, Stage A (docs/cf3g_v2_paired_proposal.md; user-approved 2026-11-28). Each arm = frozen
CF-3e (cf3e_long_hf*) + ONE change: the student trains on records its six teachers never saw, instead of records they trained on.
Teachers, alpha, model, steps, seeds, selection, frame, family shares: unchanged.

    leap : the 48 LEAP training hands use rows 417-834 instead of 0-417 (CF2 trained all of 0-417; D1-bv rows 0-48).
           Stage-0 dose (teacher_dose_v1): fresh - memorised six-teacher target corr +0.44 mm [0.38, 0.50] -> proceed.
    body : generated bodies P0000-P0799 use rows 25-50 instead of 0-25 (D1-bv trained 0-25; CF2 trained 0-400 of P0000-P0049
           only, so those 50 stay memorised by CF2, as in CF-3e). Stage-0 dose +0.13 mm [0.07, 0.19] -> proceed (marginal).
New rows come from the same sampler and seed plan as the old ones (iid), so the data distribution is unchanged; only whether
the teachers saw those exact records changes. Asserted: only `train` row ranges (+ out_dir, _note) differ from CF-3e; new
rows do not overlap any validation rows (LEAP 900-942 of the training hands; bodies 410-420 of P0000-P0049).

READOUT (fixed here, before launch; implemented in scripts/decoder/benchmark/cf3m_readout.py, committed before any result).
Per arm, per seed: arm - CF-3e of the same seed (CF-3e re-scored in the same benchmark jobs), paired bootstrap over scored
records, seeds never pooled. Pools: source val, 105 pool, 360 pool, held-out bodies P0800-P0849 (training-log eval at the
selected checkpoint; plus by rule-E frame band where per-record data exist), Wuji, Shadow, 7 external hands (flip-hand /
MetaHand rows reported separately where applicable); corr_mm, wrist_mm, corr_success, median, p90.
  * new-family score per seed = mean of the per-hand corr_mm differences over Wuji, Shadow and the 7 external hands (9 hands).
  * noise anchors (CF-3e seed spread): 105 0.20, Wuji 0.54, Shadow 0.82, held-out bodies 0.14 mm; external hands: per-hand
    paired 95 % intervals.
  * "memorisation hurts the student": new-family score < 0 in >= 2/3 seeds AND significantly better (95 % CI upper < 0) on
    >= 5 of the 9 new-family hands in >= 2/3 seeds, AND not worse than CF-3e on source val / 105 pool by more than the anchors
    (point estimate) in any seed.
  * "memorisation helps": the mirror image (score > 0 in >= 2/3 seeds and significantly worse on >= 5 of 9 hands in >= 2/3).
  * otherwise "no detectable effect".
  * Stage B (paired fold students) is launched only on the user's go-ahead, and proposed only if an arm reads "hurts" or "helps".
Interim training logs are descriptive only.

    python scripts/decoder/dlevels/make_cf3m.py
"""
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SC = REPO / "configs/decoder/scratch"
BM = REPO / "configs/decoder/benchmark"
ARMS = {"leap": (417, 834), "body": (25, 50)}
SFX = ("", "_s1", "_s2")


def is_prim(t):
    return "/prim/" in t.get("npz", "")


def is_leap(t):
    return t["kind"] == "leap" and not is_prim(t)


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def make(arm, src, sfx):
    c = copy.deepcopy(src)
    lo, hi = ARMS[arm]
    n = 0
    for t in c["train"]:
        if arm == "leap" and is_leap(t):
            assert t["rows"] == [0, 417]; t["rows"] = [lo, hi]; n += 1
        elif arm == "body" and is_prim(t):
            assert t["rows"] == [0, 25]; t["rows"] = [lo, hi]; n += 1
    assert n == (48 if arm == "leap" else 800), (arm, n)
    tr = {t["npz"] for t in c["train"] if (is_leap(t) if arm == "leap" else is_prim(t))}
    for v in c["val"] + [x for xs in c["extra_eval"].values() for x in xs]:
        if v["npz"] in tr and "rows" in v:
            assert not overlaps((lo, hi), v["rows"]), (arm, v)
    c["out_dir"] = f"${{GETKITE_RUNS}}/cf3m_{arm}_hf{sfx}"
    c["_note"] = (f"CF-3m-{arm} (2026-11-28): frozen CF-3e (cf3e_long_hf{sfx}) + one change (student trains on teacher-unseen "
                  f"rows {lo}-{hi}); docs/cf3g_v2_paired_proposal.md. Made by scripts/decoder/dlevels/make_cf3m.py.")
    diff = {k for k in set(src) | set(c) if src.get(k) != c.get(k)} - {"_note", "out_dir"}
    assert diff == {"train"}, (arm, diff)
    st = [t for t in src["train"]]; ct = [t for t in c["train"]]
    assert len(st) == len(ct) and all({k: v for k, v in a.items() if k != "rows"} == {k: v for k, v in b.items() if k != "rows"}
                                      for a, b in zip(st, ct))
    return c


def bench(arm):
    for suf in ("", "_leapval105", "_leap360"):
        b = json.loads((BM / f"transfer_v1_cf3flong{suf}.json").read_text())
        sy = {}
        for s in range(3):
            sx = "" if s == 0 else f"_s{s}"
            for k in ("cf2", "d1bv", "cf3e"):
                sy[f"{k}_s{s}"] = b["systems"][f"{k}_s{s}"]
            sy[f"cf3m{arm}_s{s}"] = {"kind": "j1", "ckpt": f"${{GETKITE_RUNS}}/cf3m_{arm}_hf{sx}/best.pt"}
        b["systems"] = sy
        b["out_dir"] = f"${{GETKITE_RUNS}}/transfer_benchmark_v1_cf3m{arm}{suf}"
        b["_note"] = f"CF-3m-{arm} vs CF-3e (re-scored in the same job), CF2, D1-bv; docs/cf3g_v2_paired_proposal.md."
        (BM / f"transfer_v1_cf3m{arm}{suf}.json").write_text(json.dumps(b, indent=1))


def main():
    for sfx in SFX:
        src = json.loads((SC / f"cf3e_long_hf{sfx}.json").read_text())
        frozen = REPO.parent / "getkite_runs/frozen_cf3e_2026-11-27/configs" / f"cf3e_long_hf{sfx}.json"
        if frozen.exists():
            assert json.loads(frozen.read_text()) == src, f"cf3e_long_hf{sfx}.json differs from the frozen copy"
        for arm in ARMS:
            c = make(arm, src, sfx)
            (SC / f"cf3m_{arm}_hf{sfx}.json").write_text(json.dumps(c, indent=1))
            print(f"cf3m_{arm}_hf{sfx or '_s0'}: train rows changed for {arm}; entries {len(c['train'])}")
    for arm in ARMS:
        bench(arm)
        print(f"benchmark configs transfer_v1_cf3m{arm}{{,_leapval105,_leap360}}.json")


if __name__ == "__main__":
    main()
