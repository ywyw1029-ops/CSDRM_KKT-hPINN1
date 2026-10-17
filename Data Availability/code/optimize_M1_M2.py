# -*- coding: utf-8 -*-
"""NSGA-II optimization on the KKT-hPINN surrogate (LCOH and net CO2 emissions).
Defaults reproduce M2-3: TAU=0.95, feasibility probability = minimum of the two classifiers,
population 100, 200 generations, coke yield <= 0.01 %, SEED=20260622.
M2-1 / M2-2: TAU=0.5 / TAU=0.8.  M1: TAU=0 (no feasibility constraint), SEED=20260615.
Aspen Plus validation (VALIDATE=1) uses the model in aspen/ and requires an Aspen Plus COM interface
(CodeLibrary_260618.Simulation, not included). Output: results/M2_rerun/."""
import os, sys, time, json
import numpy as np, pandas as pd, joblib
import torch, torch.nn as nn, torch.nn.functional as F
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
CODE=os.path.dirname(os.path.abspath(__file__)); HERE=os.path.dirname(CODE); CSDRM=HERE; sys.path.append(CODE); sys.path.append(os.path.join(HERE,"aspen"))
from tea_lca import calculate_lcoh, calculate_lca, SIGN_RULES
try: from CodeLibrary_260618 import Simulation   # Aspen COM wrapper (not included)
except ModuleNotFoundError: Simulation=None
MODELS=os.path.join(HERE, os.environ.get("MODELS","models")); RES=os.path.join(HERE,"results")
INPUT_DIM,HID,HN=6,128,3; SEED=int(os.environ.get("SEED","20260622"))
TAU=float(os.environ.get("TAU","0.95")); POP=int(os.environ.get("POP","100")); GEN=int(os.environ.get("GEN","200"))
COKE_MAX=float(os.environ.get("COKE_MAX","0.01"))   # coke yield limit [%]
VALIDATE=os.environ.get("VALIDATE","0")=="1"; RECYCLE=int(os.environ.get("RECYCLE","999"))
COLS=["Temperature","Pressure","Oxidant_Methane_Ratio","H2O_CO2_Ratio","Total_Feed_Rate","NG_Feed_Rate"]
meta=json.load(open(os.path.join(MODELS,"surrogate_meta.json"),encoding="utf-8"))
OUT_COLS=meta["output_cols"]; Z0=meta["z0_dim"]; MASS=meta["mass_balance"]; ENB=meta.get("energy_balance",{})
print(f"[meta] {meta['schema']} z0={Z0} energy={meta['energy_mode']} B={meta['B_rows']}  TAU={TAU} POP={POP} GEN={GEN} VALIDATE={VALIDATE}",flush=True)
maxabs=np.load(os.path.join(MODELS,"surrogate_maxabs.npy")); inmax=maxabs[:INPUT_DIM]; outmax=maxabs[INPUT_DIM:INPUT_DIM+Z0]
def brow(spec):
    r=np.zeros(Z0)
    for c,s in spec.items():
        if c in OUT_COLS: r[OUT_COLS.index(c)]=s
    return r
rows=[brow(MASS)*outmax]
if "energy" in meta["B_rows"]: rows.append(brow(ENB)*outmax)
_B=torch.tensor(np.vstack(rows),dtype=torch.float32); _A=torch.zeros((_B.shape[0],INPUT_DIM)); _b=torch.zeros((_B.shape[0],1))
class NNOPT(nn.Module):
    def __init__(s,idim,hdim,hnum,zdim,A,B,b):
        super().__init__(); chunk=torch.mm(B.t(),torch.inverse(torch.mm(B,B.t())))
        Astar=-torch.mm(chunk,A); Bstar=torch.eye(zdim)-torch.mm(chunk,B); bstar=torch.matmul(chunk,b).squeeze(-1)
        s.layers=nn.ModuleList([nn.Linear(idim,hdim)])
        for _ in range(hnum-1): s.layers.append(nn.Linear(hdim,hdim))
        s.layers.append(nn.Linear(hdim,zdim))
        s.fc_fixed1=nn.Linear(zdim,zdim,bias=False); s.fc_fixed1.weight=nn.Parameter(Bstar,requires_grad=False)
        s.fc_fixed2=nn.Linear(idim,zdim,bias=False); s.fc_fixed2.weight=nn.Parameter(Astar,requires_grad=False); s.fc_fixed2.bias=nn.Parameter(bstar,requires_grad=False)
    def forward(s,x):
        x0=x
        for l in s.layers[:-1]: x=F.relu(l(x))
        return s.fc_fixed1(s.layers[-1](x))+s.fc_fixed2(x0)
_model=NNOPT(INPUT_DIM,HID,HN,Z0,_A,_B,_b)
_model.load_state_dict(torch.load(os.path.join(MODELS,"CSDRM_surrogate_sd.pth"),map_location="cpu",weights_only=True)); _model.eval()
def predict_batch(X):
    with torch.no_grad(): y=_model(torch.tensor(np.asarray(X,np.float64)/inmax,dtype=torch.float32)).numpy()
    return y*outmax
_clfpath=os.environ.get("CLF_BUNDLE", os.path.join(MODELS,"feasibility_clf_bundle.joblib"))
_bd=joblib.load(_clfpath); _sc=_bd["scaler"]; _clf=_bd["clf"]; _ci=list(_clf.classes_).index(1)
_clf2=None; _clf2path=os.environ.get("CLF_BUNDLE2", os.path.join(MODELS,"feasibility_clf_bundle_AL.joblib"))  # min(clf, clf2); CLF_BUNDLE2="" = baseline classifier only
if _clf2path:
    _bd2=joblib.load(_clf2path); _sc2=_bd2["scaler"]; _clf2=_bd2["clf"]; _ci2=list(_clf2.classes_).index(1)
    print(f"[clf] ENSEMBLE min({os.path.basename(_clfpath)}, {os.path.basename(_clf2path)})",flush=True)
else:
    print(f"[clf] {os.path.basename(_clfpath)}",flush=True)
def pfeas(X):
    X=np.asarray(X,np.float64); p=_clf.predict_proba(_sc.transform(X))[:,_ci]
    if _clf2 is not None: p=np.minimum(p,_clf2.predict_proba(_sc2.transform(X))[:,_ci2])
    return p
TMAX=float(os.environ.get("TMAX","1000"))   # upper temperature bound [°C]
XL=np.array([500.,1.02,1.0,1.0,2000.,0.0]); XU=np.array([TMAX,7.0,3.0,5.0,5000.,3000.]); _sign=[(k,s) for k,s in SIGN_RULES.items()]
class Prob(Problem):
    def __init__(s): super().__init__(n_var=6,n_obj=2,n_constr=4,xl=XL,xu=XU,type_out=np.double)
    def _evaluate(s,x,out,*a,**k):
        n=len(x); f=np.full((n,2),1e6); g=np.zeros((n,4)); pf=pfeas(x); Y=predict_batch(x); g[:,3]=TAU-pf
        for i in range(n):
            if pf[i]<TAU: g[i,:3]=1e6; continue
            y=Y[i]
            if any((sg>0 and y[kk]<0) or (sg<0 and y[kk]>0) for kk,sg in _sign if kk<len(y)): g[i,:3]=1e6; continue
            g[i,0]=y[0]-1500; g[i,1]=(-1e-6 if y[1]==0 else (y[1]/(y[2]+y[3]))*100-COKE_MAX); g[i,2]=-y[4]+1e-6
            if any(g[i,:3]>0): continue
            try:
                L,TE=calculate_lcoh(y[:54]); E=calculate_lca(y[:54],TE)
                if np.all(np.isfinite([np.real(L),E])): f[i,0]=float(np.real(L)); f[i,1]=float(E)
            except Exception: pass
        out["F"]=np.nan_to_num(f,nan=1e6,posinf=1e6,neginf=1e6); out["G"]=np.nan_to_num(g,nan=-1e-6,posinf=1e6,neginf=1e6)
print(f"[search] EB M2 ...",flush=True)
res=minimize(Prob(),NSGA2(pop_size=POP,crossover=SBX(prob=0.9,eta=10),mutation=PM(prob=1/6,eta=10),eliminate_duplicates=True),get_termination("n_gen",GEN),seed=SEED,verbose=False)
Fr=np.atleast_2d(res.F); X=np.atleast_2d(res.X); v=Fr[:,0]<1e5; Fr=Fr[v]; X=X[v]
o=np.argsort(Fr[:,0]); Fr=Fr[o]; X=X[o]; keep=[]; best=np.inf
for j in range(len(Fr)):
    if Fr[j,1]<best-1e-12: keep.append(j); best=Fr[j,1]
Fr=Fr[keep]; X=X[keep]; T=X[:,0]
d=pd.DataFrame(X,columns=COLS); d["LCOH"]=Fr[:,0]; d["Total_Emission"]=Fr[:,1]
print(f"[front] {len(d)} pts  LCOH {Fr[:,0].min():.3f}~{Fr[:,0].max():.3f} E {Fr[:,1].min():.3f}~{Fr[:,1].max():.3f}  NG {d.NG_Feed_Rate.min():.0f}~{d.NG_Feed_Rate.max():.0f}  T<999:{int((T<999).sum())} T=1000:{int((T>=999.5).sum())}",flush=True)
_tag="" if TMAX>=1000 else f"_Tmax{int(TMAX)}"   # separate file name for capped runs
_otag=os.environ.get("OUTTAG","")                # optional output tag
os.makedirs(os.path.join(RES,"M2_rerun"),exist_ok=True); out=os.path.join(RES,"M2_rerun",f"M2EB_explored_validated_tau{TAU}{_tag}{_otag}_s{SEED}.csv")
if not VALIDATE:
    d["converged"]=None; d.to_csv(out,index=False,encoding="utf-8-sig"); print("[no-validate] saved",out,flush=True); sys.exit(0)
def make_sim(tries=4):
    last=None
    for k in range(tries):
        try:
            s=Simulation(AspenFileName="Simplified_CSDRM_260529.bkp",WorkingDirectoryPath=os.path.join(CSDRM,"aspen"),VISIBILITY=False); s.EngineRun(); return s
        except Exception as e: last=e; time.sleep(8*(k+1))
    raise last
def aspen_run(sim,r):
    sim.DialogSuppression(True); sim.EngineReinit()
    sim.BLK_Temp("R-1",float(r.Temperature)); sim.BLK_Pres("R-1",float(r.Pressure))
    sim.FO_Design_Spec("OXI-M",float(r.Oxidant_Methane_Ratio)); sim.FO_Design_Spec("H2O-CO2",float(r.H2O_CO2_Ratio))
    sim.STRM_CH4_M_INPUT("CH4",float(r.Total_Feed_Rate)); sim.STRM_TFlow_Input("NG",float(r.NG_Feed_Rate))
    sim.AspenSimulation.Engine.Run2()
    return sim.AspenSimulation.Tree.Elements("Data").Elements("Results Summary").Elements("Run-Status").Elements("Output").Elements("PER_ERROR").Value==0
print("Aspen launching...",flush=True); sim=make_sim(); print("Aspen ready",flush=True)
conv=[]; nconv=0
for i in range(len(d)):
    if i>0 and i%RECYCLE==0:
        try: sim.CloseAspen()
        except Exception: pass
        time.sleep(3)
        try: sim=make_sim()
        except Exception: pass
    r=d.iloc[i]; c=None
    for att in range(2):
        try: c=aspen_run(sim,r); break
        except Exception:
            try: sim.CloseAspen()
            except Exception: pass
            time.sleep(4)
            try: sim=make_sim()
            except Exception: pass
    conv.append(c); nconv+=(c is True)
    if (i+1)%10==0: print(f"  [{i+1}/{len(d)}] feas={nconv}",flush=True); d.assign(converged=conv+[None]*(len(d)-len(conv))).to_csv(out,index=False,encoding="utf-8-sig")
try: sim.CloseAspen()
except Exception: pass
d["converged"]=conv; d.to_csv(out,index=False,encoding="utf-8-sig")
print(f"\n==== EB M2 tau={TAU}: Aspen converged {nconv}/{len(d)} ({nconv/max(len(d),1):.0%}) ====  saved {out}",flush=True)
