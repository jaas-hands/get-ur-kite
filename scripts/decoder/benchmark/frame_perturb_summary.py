"""Step 0 readout (docs/cf_frame_jitter_proposal.md): CF2 contact error vs a deliberate rotation of its canonical frame.

Predeclared: causal if, in each seed, the mean increase (over the 6 axis/sign directions) rises monotonically with the angle
and the 10 deg increase is >= 0.3 mm, in 3/3 seeds. Hands: held-out LEAP with canonical frame < 3 deg from the mean.
Also shown for comparison: the observed CF2 - D1-long gap by natural frame angle (leapval105_angle.txt).
"""
import json
import os

import numpy as np

R = os.environ["GETKITE_RUNS"]
d = json.load(open(f"{R}/cf2_frame_perturb/frame_perturb.json"))
ANG = [2.5, 5, 10, 15]
ok_all = True
for n in sorted(d["mean"]):
    M = d["mean"][n]; base = M["none+0"]
    rise = {a: np.mean([M[f"{ax}{s}{a:g}"]["corr_mm"] - base["corr_mm"] for ax in "nad" for s in "+-"]) for a in ANG}
    byax = {a: {ax: np.mean([M[f"{ax}{s}{a:g}"]["corr_mm"] - base["corr_mm"] for s in "+-"]) for ax in "nad"} for a in ANG}
    wr = {a: np.mean([M[f"{ax}{s}{a:g}"]["wrist_deg"] - base["wrist_deg"] for ax in "nad" for s in "+-"]) for a in ANG}
    cs = {a: np.mean([M[f"{ax}{s}{a:g}"]["compat_success"] - base["compat_success"] for ax in "nad" for s in "+-"]) for a in ANG}
    mono = all(rise[ANG[i]] < rise[ANG[i + 1]] for i in range(len(ANG) - 1)) and rise[ANG[0]] > 0
    ok = mono and rise[10] >= 0.3; ok_all &= ok
    print(f"{n}: base corr {base['corr_mm']:.3f}")
    for a in ANG:
        print(f"  {a:4g} deg: corr +{rise[a]:.3f} mm (n {byax[a]['n']:+.3f}, a {byax[a]['a']:+.3f}, d {byax[a]['d']:+.3f})"
              f"  wrist_deg {wr[a]:+.3f}  compat_success {cs[a]:+.3f}")
    print(f"  monotonic {mono}; 10 deg increase {rise[10]:.3f} >= 0.3: {rise[10] >= 0.3} -> {'causal' if ok else 'not met'}")
print(f"\nStep 0 rule (3/3 seeds): {'MET: frame error is causal; Step 1 (CF-J) is supported' if ok_all else 'NOT MET'}")
