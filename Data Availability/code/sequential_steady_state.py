# -*- coding: utf-8 -*-
"""Sequential steady-state optimization: M2-3 re-optimized under varying natural gas price,
electricity price, CO selling price and carbon tax (surrogate only; population 100, 200 generations,
coke yield <= 0.01 %). MODE=verify (default) runs the base case only; MODE=sweep runs all four sweeps.
Output: results/M2_rerun/."""
import os, sys, math, json
import numpy as np, pandas as pd, joblib
import torch, torch.nn as nn, torch.nn.functional as F
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
CODE=os.path.dirname(os.path.abspath(__file__)); HERE=os.path.dirname(CODE); CSDRM=HERE; sys.path.append(CODE)
from tea_lca import adj, calculate_lca, SIGN_RULES
MODELS=os.path.join(HERE,"models"); RES=os.path.join(HERE,"results")
INPUT_DIM,HID,HN=6,128,3
SEED=int(os.environ.get("SEED","20260622")); TAU=float(os.environ.get("TAU","0.95"))
POP=int(os.environ.get("POP","100")); GEN=int(os.environ.get("GEN","200"))
COKE_MAX=float(os.environ.get("COKE_MAX","0.01"))   # coke yield limit [%]
MODE=os.environ.get("MODE","verify")
# base prices matching the M2-3 generator's calculate_lcoh defaults
BASE_NG=199.12; BASE_CO=300.0; BASE_ELEC=0.115; SEC_RATIO=0.09/0.115; BASE_TAX=0.0
COLS=["Temperature","Pressure","Oxidant_Methane_Ratio","H2O_CO2_Ratio","Total_Feed_Rate","NG_Feed_Rate"]
meta=json.load(open(os.path.join(MODELS,"surrogate_meta.json"),encoding="utf-8"))
OUT_COLS=meta["output_cols"]; Z0=meta["z0_dim"]; MASS=meta["mass_balance"]; ENB=meta.get("energy_balance",{})
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
_clf2path=os.environ.get("CLF_BUNDLE2", os.path.join(MODELS,"feasibility_clf_bundle_AL.joblib"))
_bd2=joblib.load(_clf2path); _sc2=_bd2["scaler"]; _clf2=_bd2["clf"]; _ci2=list(_clf2.classes_).index(1)
print(f"[clf] ENSEMBLE min({os.path.basename(_clfpath)}, {os.path.basename(_clf2path)})  TAU={TAU} POP={POP} GEN={GEN}",flush=True)
def pfeas(X):
    X=np.asarray(X,np.float64); p=_clf.predict_proba(_sc.transform(X))[:,_ci]
    return np.minimum(p,_clf2.predict_proba(_sc2.transform(X))[:,_ci2])
XL=np.array([500.,1.02,1.0,1.0,2000.,0.0]); XU=np.array([1000.,7.0,3.0,5.0,5000.,3000.]); _sign=[(k,s) for k,s in SIGN_RULES.items()]
def lcoh_priced(y,P_NG,P_CO,P_elec):
    rF1=adj(3*8000*1/1e6,2002); rCP1=adj(5840*(y[5])**0.82/1e6,1988)
    rR1=adj(math.ceil(y[6]/(6000*0.1*3.14*((0.1/2-0.008)**2)*12))*3500/1e6,2002)
    rF2=adj(3*8000*1/1e6,2002); rCP2=adj(5840*(y[7])**0.82/1e6,1988)
    TI=y[8];TO=y[9];HD=y[10]*(-1000)
    rE1=adj(25000*(((HD/(562.5*(((TI-32)-(TO-37))/math.log((TI-32)/(TO-37)))))/9.29)**0.6)/1e6,2002)
    rF3=adj(3*8000*1/1e6,2002); rPSA1=adj(1989043*((y[11]/1000)**0.7)/1e6,2021)
    rHX1=adj(20000*((y[12]/100)**0.6)/1e6,2002); rR2=adj(7e6*((-y[13])/1e5)**0.6/1e6,2002)
    rHX2=adj(20000*((y[14]/100)**0.6)/1e6,2002); rHX3=adj(20000*((y[15]/100)**0.6)/1e6,2002)
    VF=y[16];H=y[17]
    rP1=adj((2*(math.exp(5.1029-1.2217*math.log(4.403*VF*math.sqrt(0.3048*H))+0.0771*(math.log(4.403*VF*math.sqrt(0.3048*H)))**2))
            *(3*math.exp(8.833-0.6019*math.log(4.403*VF*math.sqrt(0.3048*H))+0.0519*(math.log(4.403*VF*math.sqrt(0.3048*H)))**2)))/1e6,2012)
    TPEC=rF1+rCP1+rR1+rF2+rCP2+rE1+rF3+rPSA1+rHX1+rR2+rHX2+rHX3+rP1
    Deliv=1.1*TPEC; Direct=Deliv*(1+0.47+0.36+0.68+0.11+0.18+0.1+0.7); Indir=Deliv*(0.33+0.41+0.04+0.22+0.44); FCI_U=Direct+Indir
    AmCO=y[19]*8; AV_CO=AmCO*P_CO/1e6
    AmNG=(y[20]+y[21])*8; AmBFW=y[22]*8; AmCOcap=y[23]*8000; AmCO2cap=y[24]*8; AmCO2=(y[25]-y[26])*8
    TAC=(AmNG*P_NG/1e6+AmBFW*0.15/1e6+12.9688*25000/1e6+AmCOcap*7.868*0.5122/1e6+AmCO2cap*40/1e6+AmCO2*70/1e6)
    AOLC=(5*3*37*8*365*(8000/8760)*1.928930279)/1e6
    TE=y[27]*8000+y[28]*12.63*8000+y[29]*8000+y[30]*8000
    Util=TE*P_elec/1e6+(y[31]*0.172)*(P_elec*SEC_RATIO)/1e6
    Maint=0.05*FCI_U; OpSup=0.15*AOLC; Var=TAC+AOLC+OpSup+Util+Maint+0.15*Maint
    TPC=((TAC+AOLC+OpSup+Util+Maint+0.15*Maint+0.15*AOLC+0.01*FCI_U+0.01*FCI_U+0.6*(AOLC+OpSup+Maint)+0.1*(AOLC+OpSup+Maint))/(1-0.05-0.05))
    OpR=np.array([0.5,0.9]+[1]*23); tbp=np.sum(AV_CO*OpR); tpcs=np.sum(Var*(1-OpR)-TPC)
    CRF=(0.07*1.07**25)/(1.07**25-1); SU=FCI_U*CRF*24.4*0.1; AnH2=y[32]*8
    return ((FCI_U*CRF*25-tpcs-tbp+SU)/(AnH2*24.4))*1e3, TE
class Prob(Problem):
    def __init__(s,P_NG,P_CO,P_elec,tax):
        super().__init__(n_var=6,n_obj=2,n_constr=4,xl=XL,xu=XU,type_out=np.double)
        s.P_NG=P_NG;s.P_CO=P_CO;s.P_elec=P_elec;s.tax=tax
    def _evaluate(s,x,out,*a,**k):
        n=len(x); f=np.full((n,2),1e6); g=np.zeros((n,4)); pf=pfeas(x); Y=predict_batch(x); g[:,3]=TAU-pf
        for i in range(n):
            if pf[i]<TAU: g[i,:3]=1e6; continue
            y=Y[i]
            if any((sg>0 and y[kk]<0) or (sg<0 and y[kk]>0) for kk,sg in _sign if kk<len(y)): g[i,:3]=1e6; continue
            g[i,0]=y[0]-1500; g[i,1]=(-1e-6 if y[1]==0 else (y[1]/(y[2]+y[3]))*100-COKE_MAX); g[i,2]=-y[4]+1e-6
            if any(g[i,:3]>0): continue
            try:
                L,TE=lcoh_priced(y[:54],s.P_NG,s.P_CO,s.P_elec); E=calculate_lca(y[:54],TE); L=float(np.real(L)); E=float(E)
                if np.isfinite(L) and np.isfinite(E): f[i,0]=L+E*s.tax/1000.0; f[i,1]=E
            except Exception: pass
        out["F"]=np.nan_to_num(f,nan=1e6,posinf=1e6,neginf=1e6); out["G"]=np.nan_to_num(g,nan=-1e-6,posinf=1e6,neginf=1e6)
def get_front(P_NG,P_CO,P_elec,tax):
    res=minimize(Prob(P_NG,P_CO,P_elec,tax),NSGA2(pop_size=POP,crossover=SBX(prob=0.9,eta=10),mutation=PM(prob=1/6,eta=10),eliminate_duplicates=True),get_termination("n_gen",GEN),seed=SEED,verbose=False)
    F=np.atleast_2d(res.F); X=np.atleast_2d(res.X); v=F[:,0]<1e5; F=F[v]; X=X[v]
    o=np.argsort(F[:,0]); F=F[o]; X=X[o]; keep=[]; best=np.inf
    for j in range(len(F)):
        if F[j,1]<best-1e-12: keep.append(j); best=F[j,1]
    return F[keep],X[keep]
def extremes(F,X,tag):
    L=F[:,0]; E=F[:,1]; ci=int(np.argmin(L)); ei=int(np.argmin(E))
    ln=(L-L.min())/np.ptp(L); en=(E-E.min())/np.ptp(E); ki=int(np.argmin(ln**2+en**2))
    print(f"[{tag}] n={len(F)}")
    for nm,ix in [("cost-opt",ci),("knee",ki),("emiss-opt",ei)]:
        print(f"   {nm:9s}: L={L[ix]:.3f} E={E[ix]:.3f} OM={X[ix,2]:.2f} H2O/CO2={X[ix,3]:.2f}")

if MODE=="verify":
    FR,X=get_front(BASE_NG,BASE_CO,BASE_ELEC,BASE_TAX)
    extremes(FR,X,"VERIFY base (tax=0)")
    print("target ENS: cost 2.695/7.231 OM1.47 | emiss 3.414/2.999 OM2.79")
    sys.exit(0)

SWEEPS=[("tax",np.linspace(0,180,13)),("CO",np.linspace(200,500,12)),
        ("NG",np.linspace(100,320,12)),("elec",np.round(np.linspace(0.108,0.190,12),4))]
for key,vals in SWEEPS:
    if os.environ.get("ONLY","") and os.environ["ONLY"]!=key: continue
    os.makedirs(os.path.join(RES,"M2_rerun"),exist_ok=True); fp=os.path.join(RES,"M2_rerun",f"qdENS_sweep_{key}_full.csv")
    if os.path.exists(fp): os.remove(fp)
    for v in vals:
        P_NG,P_CO,P_elec,tax=BASE_NG,BASE_CO,BASE_ELEC,BASE_TAX
        if key=="tax": tax=float(v)
        elif key=="CO": P_CO=float(v)
        elif key=="NG": P_NG=float(v)
        elif key=="elec": P_elec=float(v)
        FR,X=get_front(P_NG,P_CO,P_elec,tax)
        base_L=FR[:,0]-FR[:,1]*tax/1000.0
        df=pd.DataFrame(np.column_stack([X,FR[:,0],base_L,FR[:,1],np.full(len(FR),v)]),
                        columns=COLS+["LCOH_tax","LCOH_base","Net_emission",key])
        df.to_csv(fp,mode="a",index=False,header=not os.path.exists(fp))
        print(f"  {key}={v:.4g}: n={len(FR)} L {FR[:,0].min():.3f}~{FR[:,0].max():.3f} E {FR[:,1].min():.2f}~{FR[:,1].max():.2f}",flush=True)
    print(f"[saved] {fp}",flush=True)
