"""Does KITE's 256-point expansion change psi's z vs our raw 2-6 point input? (repetition only, no jitter, + 1mm jitter)"""
import torch
def enc(psi,P,N,V,bs=100):
    return torch.cat([psi(P[i:i+bs],N[i:i+bs],V[i:i+bs]) for i in range(0,len(P),bs)])
from getkite.decoder.config import Config
from getkite.decoder.data import SplitData
from getkite.decoder.kite_release import KitePsi, load_release
from getkite.decoder import paths
K=paths.get("KITE_ROOT")
cfg=Config(data_dir=paths.get("GETKITE_DATASET"), regime="kite", frame="world")
rel=load_release(f"{K}/checkpoints/decoder/release.pt", K)
psi=KitePsi(rel.modules["contact_only_encoder"]).eval()
d=SplitData(cfg,"val",max_examples=1000); b=d.batch(torch.arange(d.n))
p,n,v=b["c_pos"],b["c_normal"],b["c_valid"]
def expand(p,n,v,jit=0.0,T=256,g=torch.Generator().manual_seed(0)):
    B=len(p); P=torch.zeros(B,T,3); N=torch.zeros(B,T,3)
    for i in range(B):
        idx=torch.nonzero(v[i]).view(-1); k=len(idx)
        cnt=torch.full((k,),T//k); cnt[torch.randperm(k,generator=g)[:T%k]]+=1
        sel=idx.repeat_interleave(cnt); sel=sel[torch.randperm(T,generator=g)]
        P[i]=p[i,sel]+jit*torch.randn(T,3,generator=g); N[i]=n[i,sel]
    return P,N,torch.ones(B,T,dtype=torch.bool)
with torch.no_grad():
    z0=psi(p,n,v)
    for jit in (0.0,0.001,0.003):
        z1=enc(psi,*expand(p,n,v,jit))
        rel_err=((z1-z0).norm(dim=1)/z0.norm(dim=1))
        cos=torch.nn.functional.cosine_similarity(z0,z1,dim=1)
        # how big is that vs typical distance between different examples' z?
        dz_between=(z0[1:]-z0[:-1]).norm(dim=1)/z0[:-1].norm(dim=1)
        print(f"jitter {jit*1000:.0f}mm: |z_exp - z_raw|/|z_raw| median {rel_err.median():.4f} p90 {rel_err.quantile(.9):.4f}; cos median {cos.median():.5f}; between-example rel dist median {dz_between.median():.4f}")
    z1=enc(psi,*expand(p,n,v,0.0))
    for name,z in (("raw",z0),("expanded",z1)):
        s=torch.linalg.svdvals(z-z.mean(0)).pow(2); pr=s/s.sum()
        print(f"{name}: participation ratio {float(1/(pr**2).sum()):.1f}/256, |z| {z.norm(dim=1).mean():.1f}")
