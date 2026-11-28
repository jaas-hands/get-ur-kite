"""Benchmark configs for the two extra CF-3f arms (big300k, contact), made from the CF-3f-big ones (2026-11-28).

Each scores the arm AND frozen CF-3e (plus CF2, D1-bv; reported) in the same benchmark run, on the same hardware, so the
paired gate (cf3f_gate.py) compares like with like. Pools as transfer_v1_cf3e{,_leapval105,_leap360}.json.

    python scripts/decoder/dlevels/make_cf3f_extra_bench.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
B = REPO / "configs/decoder/benchmark"


def make(arm, suffix, src_name):
    src = json.loads((B / f"{src_name}.json").read_text())
    old_sys, old_run = (("cf3fbig", "cf3f_big_hf") if "big" in src_name else ("cf3flong", "cf3f_long_hf"))
    c = dict(src)
    c["systems"] = {k.replace(old_sys, f"cf3f{arm}"): {**v, "ckpt": v["ckpt"].replace(old_run, f"cf3f_{arm}_hf")}
                    for k, v in src["systems"].items()}
    c["out_dir"] = f"${{GETKITE_RUNS}}/transfer_benchmark_v1_cf3f{arm}{suffix}"
    c["_note"] = (f"CF-3f-{arm} vs CF-3e (gate reference, scored in this same run), CF2, D1-bv; docs/cf3f_extra_arms.md. "
                  f"Made from {src_name}.json by scripts/decoder/dlevels/make_cf3f_extra_bench.py.")
    names = set(c["systems"])
    assert {f"cf3f{arm}_s{s}" for s in range(3)} <= names and {f"cf3e_s{s}" for s in range(3)} <= names, names
    assert all("cf3f_big_hf" not in v["ckpt"] and "cf3f_long_hf" not in v["ckpt"] for v in c["systems"].values())
    assert {k: v for k, v in c.items() if k not in ("systems", "out_dir", "_note")} == \
           {k: v for k, v in src.items() if k not in ("systems", "out_dir", "_note")}
    (B / f"transfer_v1_cf3f{arm}{suffix}.json").write_text(json.dumps(c, indent=1))
    print(f"transfer_v1_cf3f{arm}{suffix}: {sorted(names)}")


for arm in ("big300k", "contact"):
    make(arm, "", "transfer_v1_cf3fbig")
    make(arm, "_leapval105", "transfer_v1_cf3fbig_leapval105")
    make(arm, "_leap360", "transfer_v1_cf3flong_leap360")
