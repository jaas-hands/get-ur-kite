"""Preflight audit, part 2: morphology leakage across splits and synthetic-data distributions."""
import json, collections, numpy as np
from pathlib import Path
from kite_data.dataset import load_split
from getkite.decoder import paths
D = Path(paths.get("GETKITE_DATASET"))
S = json.load(open(D / "splits.json"))
H = {}
def hand(h):
    if h not in H: H[h] = json.load(open(D / "hands" / h / "hand.json"))
    return H[h]
def info(h):
    j = hand(h); mov = [x for x in j["joints"] if not x["locked"]]
    per = collections.Counter(x["finger"] for x in mov)
    fp = tuple(sorted((x["name"], tuple(np.round(x["origin_xyz"], 6)), tuple(x["axis"]), round(x["lower"], 4), round(x["upper"], 4), x["locked"]) for x in j["joints"]))
    return dict(topo=j["topology"], fingers=frozenset(j["fingers"]), per=tuple(sorted(per.items())), ndof=j["n_dof"],
                canon=tuple(sorted(j["dof_names"], key=int)), draw=j["geometry"]["draw"], fp=fp,
                scales=list(j["geometry"].get("length_scales", {}).values()), shifts=list(j["geometry"].get("mount_shifts_m", {}).values()))
I = {s: {h: info(h) for h in hs} for s, hs in S["hands"].items()}
tr = I["train"]
print("== 2/3/4. leakage checks")
trtopo = {v["topo"] for v in tr.values()}; trfp = {v["fp"] for v in tr.values()}
trfs = {v["fingers"] for v in tr.values()}; trper = {v["per"] for v in tr.values()}
trcanon = {v["canon"] for v in tr.values()}
for s in [k for k in I if k != "train"]:
    v = I[s].values()
    print(f"  {s:18s} hands {len(I[s]):4d} | topologies {len({x['topo'] for x in v}):3d}, shared w/ train {len({x['topo'] for x in v} & trtopo):3d}"
          f" | identical kinematics to a train hand: {sum(x['fp'] in trfp for x in v)}"
          f" | finger sets seen in train: {len({x['fingers'] for x in v} & trfs)}/{len({x['fingers'] for x in v})}"
          f" | per-finger joint-count tuples seen in train: {len({x['per'] for x in v} & trper)}/{len({x['per'] for x in v})}"
          f" | canonical-ID sets seen in train: {len({x['canon'] for x in v} & trcanon)}/{len({x['canon'] for x in v})}")
idx3 = [h for h, v in tr.items() if dict(v["per"]).get("index") == 3]
print("  train hands with a 3-joint index finger (joint-count holdout):", len(idx3))
sub = [h for h, v in tr.items() if v["fingers"] in (frozenset({"thumb", "ring"}), frozenset({"thumb", "index", "ring"}))]
print("  train hands with exactly {thumb,ring} or {thumb,index,ring}:", len(sub))
print("  test_geometry draws:", sorted({v['draw'] for v in I['test_geometry'].values()}), "| train draws:", sorted({v['draw'] for v in tr.values()}))
# geometry distance of test_geometry hands to same-topology train hands (max |joint-origin diff|, mm)
def origins(h): return np.array([x["origin_xyz"] for x in sorted(hand(h)["joints"], key=lambda x: x["name"])])
dmin = []
for h, v in I["test_geometry"].items():
    o = origins(h); dmin.append(min(np.abs(o - origins(t)).max() for t, w in tr.items() if w["topo"] == v["topo"]) * 1000)
print(f"  test_geometry: nearest same-topology train hand differs by max joint-origin offset {np.min(dmin):.2f}-{np.max(dmin):.2f} mm (median {np.median(dmin):.2f})")
print("  topologies listed per split are disjoint:", all(not (set(S['topologies'][a]) & set(S['topologies'][b]))
      for a in S['topologies'] for b in S['topologies'] if a < b and 'test_geometry' not in (a, b)),
      "| test_geometry topologies subset of train:", set(S['topologies']['test_geometry']) <= set(S['topologies']['train']))
print("\n== example files contain only their split's hands")
ex = {}
for s in S["hands"]:
    d = load_split(s, regime="kite_code", root=D / "examples" / "kite_code")
    ids = list(d["hand_ids"]); cnt = np.bincount(d["hand"], minlength=len(ids))
    ex[s] = d
    print(f"  {s:18s} hand_ids == splits.json: {sorted(ids) == sorted(S['hands'][s])} | examples {len(d['hand']):,} | per hand {cnt.min()}-{cnt.max()}")
print("  train example hands overlapping any val/test hand id:", len(set(ex['train']['hand_ids']) & set().union(*[set(ex[s]['hand_ids']) for s in ex if s != 'train'])))
print("\n== 9. distributions")
for s, v in I.items():
    vv = list(v.values())
    nd = collections.Counter(x["ndof"] for x in vv); nf = collections.Counter(len(x["fingers"]) for x in vv)
    sc = [a for x in vv for a in x["scales"]]; sh = [a * 1000 for x in vv for a in x["shifts"]]
    print(f"  {s:18s} hands {len(vv):4d}  topologies {len({x['topo'] for x in vv}):3d}  nominal(draw0) {sum(x['draw'] == 0 for x in vv):3d}"
          f"  DoF {min(nd)}-{max(nd)} (median {int(np.median([x['ndof'] for x in vv]))})  fingers/hand {dict(sorted(nf.items()))}"
          + (f"  link scale {min(sc):.3f}-{max(sc):.3f}  mount shift {min(sh):+.2f}..{max(sh):+.2f} mm" if sc else ""))
print("  DoF histogram, train:", dict(sorted(collections.Counter(x['ndof'] for x in tr.values()).items())))
for s in ["train", "val"]:
    d = ex[s]; print(f"  {s}: contacts per example k:", dict(sorted(collections.Counter(d['k'].tolist()).items())), f"| mask (moving-joint) density {d['mask'].mean():.3f}")
d = ex["val"]
from scipy.spatial.transform import Rotation as R
dp = np.linalg.norm(d["wrist_pos_init"] - d["wrist_pos_contact"], axis=1) * 1000
qa = lambda q: R.from_quat(q[:, [1, 2, 3, 0]])
ang = (qa(d["wrist_quat_init"]).inv() * qa(d["wrist_quat_contact"])).magnitude() * 180 / np.pi
print(f"  stored val start-pose perturbation: wrist offset {np.median(dp):.1f} mm median (p95 {np.percentile(dp, 95):.1f}), rotation {np.median(ang):.1f} deg median (p95 {np.percentile(ang, 95):.1f})")
print("  keys:", sorted(d.keys()))
