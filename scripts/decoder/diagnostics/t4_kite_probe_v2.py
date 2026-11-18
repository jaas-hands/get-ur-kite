"""Reference for G2: probe v2 on KITE's OWN held-out Robotiq contacts (z = stored latent_proprio), episode split."""
import json, torch
from probe_v2 import Probe, loss_fn, metrics
from getkite.decoder import paths
K=paths.get("KITE_ROOT"); dev="cuda" if torch.cuda.is_available() else "cpu"
d=torch.load(f"{K}/experiment/data/high_level_policy/object_suite_main_paper_contacts_v1_test.pt",map_location="cpu",weights_only=False)
x=d["data"]; ends=d["episode_ends"].tolist(); n=len(x["latent_proprio"])
ep=torch.zeros(n,dtype=torch.long); s=0
for i,e in enumerate(ends): ep[s:e]=i; s=e
test_eps=set(torch.randperm(len(ends),generator=torch.Generator().manual_seed(0))[:20].tolist())
te=torch.tensor([int(e) in test_eps for e in ep]); tr=~te
cp=x["contact_proprio"].float().to(dev); v=x["contact_proprio_valid_mask"].to(dev)
b={"c_pos":cp[...,:3],"c_normal":cp[...,3:],"c_valid":v}; X=x["latent_proprio"].float().to(dev)
tr,te=tr.to(dev),te.to(dev); idx=torch.nonzero(tr).view(-1)
torch.manual_seed(0); probe=Probe(X.shape[1]).to(dev); steps=20000
opt=torch.optim.AdamW(probe.parameters(),3e-4,weight_decay=1e-4); sched=torch.optim.lr_scheduler.OneCycleLR(opt,1e-3,total_steps=steps,pct_start=0.05)
g=torch.Generator().manual_seed(0)
for st in range(steps):
    i=idx[torch.randint(0,len(idx),(1024,),generator=g).to(dev)]
    loss=loss_fn(probe(X[i]),b["c_pos"][i],b["c_normal"][i],b["c_valid"][i]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if (st+1)%5000==0: print(f"step {st+1} loss {loss.item():.4g}",flush=True)
probe.eval()
print(json.dumps({name:metrics(probe,X[m],{k:t[m] for k,t in b.items()}) for name,m in (("kite_train_eps",tr),("kite_heldout_eps",te))},indent=1))
