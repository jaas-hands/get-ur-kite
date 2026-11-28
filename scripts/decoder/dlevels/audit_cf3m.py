"""CF-3m copy of audit_cf3f.py (same checks, cf3m configs). Leakage audit for the CF-3f arms (docs/cf3f_proposal.md). Fails (exit 1) if any training entry of cf3m_*_hf*.json touches:
the 105-hand held-out LEAP pool's topologies (val split), Wuji, Shadow, the external hands, held-out generated bodies
(P0800-P0899 / prim heldout), or any test_* split. Reports (does not fail) overlap with the 360-hand pool's topologies, which is
expected only for the leap arm. Also checks that no teacher checkpoint path points into a cf3f run.

    python scripts/decoder/dlevels/audit_cf3f.py
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SC = REPO / "configs/decoder/scratch"
BAD = re.compile(r"wuji|shadow|ability|inspire|xhand|schunk|metahand|sharpa|aero|/test_|/heldout/|/val/|P08\d\d", re.I)
top = lambda p: Path(p).stem.split("_")[0]  # noqa: E731
v105 = {top(t["npz"]) for t in json.loads((SC / "c3_psi50k_10x_wrist_leapval105.json").read_text())["extra_eval"]["leap_heldout"]}
p360 = {top(t["npz"]) for t in json.loads((SC / "c3_psi50k_10x_wrist_leap360.json").read_text())["extra_eval"]["leap_heldout"]}
f360 = {Path(t["npz"]).stem for t in json.loads((SC / "c3_psi50k_10x_wrist_leap360.json").read_text())["extra_eval"]["leap_heldout"]}
ok = True
files = sorted(SC.glob("cf3m_*_hf*.json"))
assert files, "no cf3f configs"
for f in files:
    c = json.loads(f.read_text())
    tr = [t.get("npz", "") for t in c["train"]]
    bad = [p for p in tr if BAD.search(p)]
    leap = [p for p in tr if "/prim/" not in p and "kite_code" in p]
    hit105 = {top(p) for p in leap} & v105
    hit360f = {Path(p).stem for p in leap} & f360
    hit360t = {top(p) for p in leap} & p360
    teach_bad = [t for t in c.get("distill_teachers", []) if "cf3f_" in t]
    fail = bool(bad or hit105 or hit360f or teach_bad)
    ok &= not fail
    print(f"{f.name:28s} {'FAIL' if fail else 'ok  '} train={len(tr)} leap_hands={len(leap)} bad={bad[:2]} val105_topologies={len(hit105)} "
          f"360_files={len(hit360f)} 360_topologies(reported)={len(hit360t)} teachers={len(c.get('distill_teachers', []))}")
sys.exit(0 if ok else 1)
