#!/usr/bin/env python3
"""Per-run performance extraction for the R6 D435 suite (tegrastats, console events, trajectories).
usage: python3 analyze_d435.py RUNS_DIR [out.json]   -- numpy only.
  RUNS_DIR: eval/results/d435_25w/runs or eval/results/d435_maxn/runs. Feeds that folder's REPORT.md."""
import re, json, sys, numpy as np
from pathlib import Path
root=Path(sys.argv[1]) / "d435" / "d435_s"
pat=re.compile(r"RAM (\d+)/(\d+)MB.*?SWAP (\d+)/.*?CPU \[([^\]]*)\] GR3D_FREQ (\d+)%.*?cpu@([\d.]+)C.*?gpu@([\d.]+)C.*?tj@([\d.]+)C.*?VDD_IN (\d+)mW.*?VDD_CPU_GPU_CV (\d+)mW.*?VDD_SOC (\d+)mW")
def teg(p):
    rows=[]
    for l in open(p):
        m=pat.search(l)
        if not m: continue
        cores=[c.split('@') for c in m.group(4).split(',')]
        u=[float(c[0].rstrip('%')) if c[0]!='off' else 0 for c in cores]
        f=[float(c[1]) if len(c)>1 else 0 for c in cores]
        rows.append(dict(ram=int(m.group(1)),swap=int(m.group(3)),cpu=u,freq=f,gr3d=int(m.group(5)),tcpu=float(m.group(6)),tgpu=float(m.group(7)),tj=float(m.group(8)),vin=int(m.group(9)),vcg=int(m.group(10)),vsoc=int(m.group(11))))
    return rows
def traj(p):
    if not p.exists(): return None
    a=np.loadtxt(p,ndmin=2)
    if len(a)==0: return None
    t=a[:,0]/1e9; x=a[:,1:4]
    return dict(n=len(a),span=t[-1]-t[0],t0=t[0],t1=t[-1],path=float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum()),endgap=float(np.linalg.norm(x[-1]-x[0])),ext=(x.max(0)-x.min(0)).tolist())
out=[]
for rd in sorted(root.glob("*/*/run*")):
    seq,arm,run=rd.parts[-3:]
    meta=json.load(open(rd/"meta.json")); ev=json.load(open(rd/"eval.json"))
    con=open(rd/"console.log",errors="ignore").read()
    T=teg(rd/"tegrastats.log")
    tot=np.array([sum(r['cpu']) for r in T]); gr=np.array([r['gr3d'] for r in T])
    # tracking window: after vocab load (RAM growth with single core) -> first sample with >=2 cores >20% or gr3d>0
    busy=[i for i,r in enumerate(T) if sum(1 for u in r['cpu'] if u>25)>=2 or r['gr3d']>5]
    i0,i1=(busy[0],busy[-1]) if busy else (0,len(T)-1)
    W=T[i0:i1+1]
    def s(k): return np.array([r[k] for r in W],float)
    cpu=np.array([r['cpu'] for r in W]); fr=np.array([r['freq'] for r in W])
    f=traj(rd/"f_run.txt"); k=traj(rd/"kf_run.txt")
    crash=("Sophus/NaN" if "SO3::exp failed" in con else "pure virtual" if "pure virtual" in con else "X error (xvfb)" if "gdk" in con or "X Window" in con else "Bus error@KF save" if "Bus error" in con else "Segfault@KF save" if "Segmentation" in con else "clean")
    kfs=[int(x) for x in re.findall(r"Map \d+ has (\d+) KFs",con)]
    out.append(dict(seq=seq,arm=arm,run=run,exit=meta.get("exit_code"),wall=meta.get("wall_s"),crash=crash,
      fails=con.count("Fail to track local map"),newmaps=con.count("Creation of new map with id"),loops=con.count("*Loop detected"),merges=con.count("Merge detected"),final_maps=ev["console"].get("maps"),kfs=kfs,
      samples=len(T),trk_s=len(W),idle_pre=i0,
      ram_peak=max(r['ram'] for r in T),ram_start=T[0]['ram'],ram_trk0=W[0]['ram'],swap_peak=max(r['swap'] for r in T),
      cpu_tot=float(cpu.sum(1).mean()),cpu_cores=cpu.mean(0).round(1).tolist(),cpu_max_core=float(cpu.max()),freq=fr.mean(0).round(0).tolist(),
      gr3d_mean=float(s('gr3d').mean()),gr3d_p95=float(np.percentile(s('gr3d'),95)),gr3d_nz=float((s('gr3d')>0).mean()),
      vin=float(s('vin').mean()),vin_pk=float(s('vin').max()),vcg=float(s('vcg').mean()),vsoc=float(s('vsoc').mean()),vin_pre=float(np.mean([r['vin'] for r in T[:max(i0,1)]])),
      tcpu=float(s('tcpu').max()),tgpu=float(s('tgpu').max()),tj=float(s('tj').max()),tj0=T[0]['tj'],
      E=float(s('vin').sum()/1000), f=f, kf=k))
json.dump(out,open(sys.argv[2] if len(sys.argv)>2 else "d435_runs.json","w"),indent=1)
