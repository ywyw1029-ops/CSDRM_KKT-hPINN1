# -*- coding: utf-8 -*-
"""Train the second feasibility classifier: the base dataset augmented with the Aspen Plus convergence
labels of points explored during optimization (data/classifier_additional_labels.csv)."""
import os, glob, joblib
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)  # repo root
DATA=os.path.join(ROOT,"data","CSDRM_simulation_dataset.csv"); M2=os.path.join(ROOT,"data","classifier_additional_labels.csv")
OUTDIR=os.environ.get("OUTDIR",os.path.join(ROOT,"retrained_models")); os.makedirs(OUTDIR,exist_ok=True)
INPUT_COLS=['Temperature','Pressure','Oxidant_Methane_Ratio','H2O_CO2_Ratio','Total_Feed_Rate','NG_Feed_Rate']
SEED=0; UP=float(os.environ.get("UPWEIGHT","30"))

# 1) base 무작위 학습데이터
df=pd.read_csv(DATA); base=df[INPUT_COLS].copy(); base["y"]=df["converged"].astype(int).values
# 2) MOO 코너 라벨 점들 수집 (모든 검증 스윕에서 converged 있는 것)
M=pd.read_csv(M2); M["y"]=M["converged"].astype(int)   # merged labels, same rows/order as the per-sweep files
print(f"MOO 코너 라벨점 {len(M)}  수렴 {M.y.mean():.2f}  T {M.Temperature.min():.0f}~{M.Temperature.max():.0f}")
Mtr,Mte=train_test_split(M,test_size=0.2,random_state=SEED,stratify=M.y)

# 3) 보강(base 가중 1, MOO 코너 가중 UP) 후 재학습
aug=pd.concat([base.assign(w=1.0), Mtr[INPUT_COLS+["y"]].assign(w=UP)],ignore_index=True)
X=aug[INPUT_COLS].values.astype(np.float64); y=aug.y.values; w=aug.w.values
sc=StandardScaler().fit(X)
clf=RandomForestClassifier(n_estimators=600,class_weight="balanced_subsample",min_samples_leaf=2,n_jobs=-1,random_state=SEED)
print(f"재학습 (UPWEIGHT={UP}, base {len(base)} + MOO_tr {len(Mtr)})...",flush=True)
clf.fit(sc.transform(X),y,sample_weight=w)
ci=list(clf.classes_).index(1)
# 4) held-out MOO 코너로 평가 (코너에서 수렴/발산 구분되나)
p=clf.predict_proba(sc.transform(Mte[INPUT_COLS].values))[:,ci]; yte=Mte.y.values
print(f"[AL] held-out 코너: pfeas|수렴 {p[yte==1].mean():.3f}  pfeas|발산 {p[yte==0].mean():.3f}  분리 {p[yte==1].mean()-p[yte==0].mean():.3f}  AUC {roc_auc_score(yte,p):.3f}")
# 고온(T>=995) 코너만
hi=Mte.Temperature.values>=995
if hi.sum()>0: print(f"[AL] T>=995 코너(n={int(hi.sum())}): pfeas|수렴 {p[hi&(yte==1)].mean():.3f}  pfeas|발산 {p[hi&(yte==0)].mean():.3f}")
out=os.path.join(OUTDIR,os.environ.get("AL_OUT","feasibility_clf_bundle_AL.joblib"))
joblib.dump({"scaler":sc,"clf":clf,"input_cols":INPUT_COLS,"recommend_thresh":0.5,"note":f"active-learning aug (UP={UP})"},out)
print("saved:",out)
