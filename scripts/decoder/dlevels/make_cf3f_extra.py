"""Two extra CF-3f arms run on Aryan's compute (2026-11-28), each made from the frozen CF-3e configs (cf3e_long_hf*), 3 seeds:

    big300k : CF-3f-big (model 256 wide x 6 layers x 8 heads) trained for 300k steps (500-step warm-up, cosine to 0 at 300k),
              i.e. the big arm with the long arm's schedule. A clean, full-length size test (no warm restart, unlike bigext).
    contact : CF-3e + a forward-kinematics contact loss on the predicted goal pose (train_joint.contact_fk_loss), weight set
              by the predeclared "auto" rule (motion / contact-FK gradient-norm ratio on the decoder at the first batch).

Both are judged like the other CF-3f arms: cf3f_gate.py against CF-3e of the same seed, plus the external hands. Each output
asserts that only the keys belonging to its change differ from CF-3e (plus _note / out_dir).

    python scripts/decoder/dlevels/make_cf3f_extra.py
"""
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SC = REPO / "configs/decoder/scratch"
CHANGED = {"big300k": {"model", "total_steps"}, "contact": {"contact_fk_weight"}}


def make(arm, src, sfx):
    c = copy.deepcopy(src)
    if arm == "big300k":
        c["model"] = dict(src["model"]) | {"d_model": 256, "n_layers": 6, "n_heads": 8}
        c["total_steps"] = 300000
    elif arm == "contact":
        c["contact_fk_weight"] = "auto"
    c["out_dir"] = f"${{GETKITE_RUNS}}/cf3f_{arm}_hf{sfx}"
    c["_note"] = (f"CF-3f-{arm} (2026-11-28): frozen CF-3e (cf3e_long_hf{sfx}) + {sorted(CHANGED[arm])}. "
                  "Made by scripts/decoder/dlevels/make_cf3f_extra.py.")
    diff = {k for k in set(src) | set(c) if src.get(k) != c.get(k)} - {"_note", "out_dir"}
    assert diff == CHANGED[arm], (arm, diff)
    return c


def main():
    for sfx in ("", "_s1", "_s2"):
        src = json.loads((SC / f"cf3e_long_hf{sfx}.json").read_text())
        big = json.loads((SC / f"cf3f_big_hf{sfx}.json").read_text())
        b300 = make("big300k", src, sfx)
        assert {k: v for k, v in b300.items() if k not in ("out_dir", "_note", "total_steps")} == \
               {k: v for k, v in big.items() if k not in ("out_dir", "_note", "total_steps")}, "big300k must equal cf3f_big except length"
        for arm, c in (("big300k", b300), ("contact", make("contact", src, sfx))):
            (SC / f"cf3f_{arm}_hf{sfx}.json").write_text(json.dumps(c, indent=1))
            print(f"cf3f_{arm}_hf{sfx or '_s0'}: changed {sorted(CHANGED[arm])}")


if __name__ == "__main__":
    main()
