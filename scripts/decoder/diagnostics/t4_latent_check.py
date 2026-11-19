"""T4 prep: does our psi reproduce KITE's stored latent_proprio on KITE's own contacts? + z stats of KITE contacts."""
import torch
from getkite.decoder.kite_release import KitePsi, load_release
from getkite.decoder import paths
K=paths.get("KITE_ROOT")
rel=load_release(f"{K}/checkpoints/decoder/release.pt", K)
psi=KitePsi(rel.modules["contact_only_encoder"]).eval()
d=torch.load(f"{K}/experiment/data/high_level_policy/object_suite_main_paper_contacts_v1_test.pt",map_location="cpu",weights_only=False)
x=d["data"]; m=d["meta"]
for k in ("dataset_xyz_translation","contact_frame","frame","latent_source","latent_proprio_source","contact_proprio_frame"):
    if k in m: print("meta",k,"=",m[k])
print("meta keys with 'latent' or 'xyz' or 'frame':",[k for k in m if any(s in k for s in ("latent","xyz","frame","transl"))])
cp=x["contact_proprio"].float(); v=x["contact_proprio_valid_mask"]; zl=x["latent_proprio"].float()
print("valid count dist:",torch.bincount(v.sum(1)).tolist())
print("contact pos range:",cp[...,:3][v].min(0).values.tolist(),cp[...,:3][v].max(0).values.tolist())
print("normal norm:",cp[...,3:][v].norm(dim=-1).mean().item())
sel=v.any(1)
with torch.no_grad():
    zo=torch.cat([psi(cp[i:i+512,:,:3],cp[i:i+512,:,3:],v[i:i+512]) for i in range(0,len(cp),512)])
e=(zo-zl)[sel].abs().max(1).values
print(f"|psi_ours(contact_proprio) - latent_proprio| max-abs per frame: median {e.median():.3e} p99 {e.quantile(.99):.3e} max {e.max():.3e} (|z|~{zl[sel].norm(dim=1).mean():.1f})")
for name,z in (("KITE latent_proprio",zl[sel]),("ours on KITE contacts",zo[sel])):
    s=torch.linalg.svdvals(z-z.mean(0)).pow(2); p=s/s.sum()
    print(f"{name}: participation ratio {float(1/(p**2).sum()):.1f}/256, |z| {z.norm(dim=1).mean():.1f}, per-dim std {z.std(0).mean():.2f}")
