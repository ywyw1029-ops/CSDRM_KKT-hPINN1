"""Train the KKT-hPINN surrogate (material and energy balances as hard constraints) on converged
samples and the baseline feasibility classifier (random forest) on all samples of
data/CSDRM_simulation_dataset.csv. Outputs are written to retrained_models/."""
import os, json
import numpy as np, pandas as pd, joblib
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root
MERGED_CSV = os.environ.get("MERGED_CSV", os.path.join(ROOT, "data", "CSDRM_simulation_dataset.csv"))
OUTDIR = os.environ.get("OUTDIR", os.path.join(ROOT, "retrained_models")); os.makedirs(OUTDIR, exist_ok=True)
TRAIN_SURROGATE = os.environ.get("TRAIN_SURROGATE", "1") == "1"

INPUT_DIM, HIDDEN_DIM, HIDDEN_NUM = 6, 128, 3
EPOCHS    = int(os.environ.get("EPOCHS", "1000"))
BATCH     = int(os.environ.get("BATCH", "64"))
LR        = 5e-5
VAL_RATIO = 0.2
SEED      = 0
CLF_KIND  = os.environ.get("CLF", "rf").lower()
MAX_ROWS  = int(os.environ.get("MAX_ROWS", "0"))
ENERGY_MODE   = os.environ.get("ENERGY_MODE", "hard").lower()    # off | soft | hard
ENERGY_LAMBDA = float(os.environ.get("ENERGY_LAMBDA", "1.0"))   # soft 모드 페널티 가중

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

INPUT_COLS = ['Temperature','Pressure','Oxidant_Methane_Ratio','H2O_CO2_Ratio','Total_Feed_Rate','NG_Feed_Rate']
# ── [MASS BALANCE] 기존 54 출력 (질량수지 hard 제약 = 맨 끝 *_INPUT/*_OUTPUT 12개) ──
OUTPUT_COLS_MB = [
    'Furnace_Temp','c106','co2_m105','ch4_m105','Q3',
    'PR_CP1','VF_R1','PR_CP2','TI_E1','TO_E1','HD_E1','VF_PSA1','HEA_HX1','HD_R2','HEA_HX2','HEA_HX3','VF_P1','H_P1',
    'Amount_H2','Amount_CO','Amount_NG_1','Amount_NG_2','Amount_BFW','Amount_CO_Capture','Amount_CO2_Capture','Amount_CO2_1','Amount_CO2_2',
    'Elec_P1_kW','Elec_PSA1_kghr','Elec_CP2_kW','Elec_CP1_kW','CW_E1_tonhr',
    'Annual_H2_Production','H2_Amount','NG_Amount_1','NG_Amount_2','BFW_Amount','CO_Amount','CO_Amount_Mole','Makeup_Amount_1','Makeup_Amount_2','DCO2_Amount',
    'CO2M_INPUT','CH4_INPUT','BFW_INPUT','NG_INPUT','AIR_INPUT',
    'H2OL1_OUTPUT','H2OL2_OUTPUT','CO2R_OUTPUT','H2OL3_OUTPUT','COV_OUTPUT','H2V_OUTPUT','EXGAS_OUTPUT',
]
# ── [ENERGY BALANCE] 추가 출력 (extract_with_failures EB 스키마와 동일 순서) ──
#  ※ R1_ENTH_GEN 제거(2026-06-20): EB 추출에서 R-1(RGibbs)의 BAL_ENTH_GEN 노드가 항상 비어
#    수렴행 전부 NaN → 학습 dropna가 전행 삭제. 경계항도 아니라 잉여 → 스키마에서 영구 제외.
ENERGY_COLS = [
    'Q_R1','Q_F1','Q_AB1','Q_PSA1','Q_VPSA1',
    'H_CO2M','H_CH4','H_BFW','H_NG','H_AIR',
    'H_H2OL1','H_H2OL2','H_CO2R','H_H2OL3','H_COV','H_H2V','H_EXGAS',
    'H_303','H_304',
]
#  soft 페널티 정규화 상수: 에너지수지 항 전형크기(Σ|항| median ≈ 1.76e6, prep 진단값).
#  실단위 잔차(~1e5)를 이걸로 나눠 무차원(상대잔차 O(0.1))으로 만들어 MSE(~1e-5)와 균형.
ENERGY_NORM = float(os.environ.get("ENERGY_NORM", "1.76e6"))

# ── 제약 행 정의 (컬럼명→부호). 인덱스가 아니라 이름으로 만들어 스키마 변동에 강건 ──
# 질량수지: 입력스트림(+) − 출력스트림(−) = 0   (항상 적용; 검증된 항등식)
MASS_BALANCE = {c: +1 for c in ['CO2M_INPUT','CH4_INPUT','BFW_INPUT','NG_INPUT','AIR_INPUT']}
MASS_BALANCE.update({c: -1 for c in ['H2OL1_OUTPUT','H2OL2_OUTPUT','CO2R_OUTPUT','H2OL3_OUTPUT','COV_OUTPUT','H2V_OUTPUT','EXGAS_OUTPUT']})
# 에너지수지: 입력엔탈피(+) + 공정에 더해진 열·일(+) − 출력엔탈피(−) = 0
#  ★검증 완료(2026-06-20, diag_energy_closure.py): flue(303/304) 포함 시 수렴데이터 상대잔차 0.00%·계수±1
#    → flue 는 경계항이 맞음(furnace 연소열원). 포함해야 닫힘 = hard 정당. (R1_ENTH_GEN 만 피처/제외)
ENERGY_BALANCE = {c: +1 for c in ['H_CO2M','H_CH4','H_BFW','H_NG','H_AIR']}
ENERGY_BALANCE.update({c: -1 for c in ['H_H2OL1','H_H2OL2','H_CO2R','H_H2OL3','H_COV','H_H2V','H_EXGAS']})
ENERGY_BALANCE.update({c: +1 for c in ['Q_R1','HD_R2','HD_E1','Q_F1','Q_AB1','Q_PSA1','Q_VPSA1',  # 블록 heat duty(공정에 +)
                                       'Elec_CP1_kW','Elec_CP2_kW','Elec_P1_kW']})                  # 압축기/펌프 일(+)
ENERGY_BALANCE.update({'H_303': -1, 'H_304': +1})   # flue: 닫는 항 (−H303+H304), diag로 ±1 확정


def build_brow(spec, cols):
    """컬럼명→부호 dict 를 출력벡터(cols 순서) 위의 계수 행으로. 없는 컬럼은 무시(경고)."""
    row = np.zeros(len(cols), dtype=np.float64); miss = []
    for c, s in spec.items():
        if c in cols: row[cols.index(c)] = s
        else: miss.append(c)
    if miss: print(f"   (행 구성 경고) cols 에 없는 항 무시: {miss}")
    return row


def make_NNOPT(zdim):
    class NNOPT(nn.Module):
        def __init__(self, idim, hdim, hnum, zdim, A, B, b):
            super().__init__()
            chunk = torch.mm(B.t(), torch.inverse(torch.mm(B, B.t())))   # B:(m,zdim) → 다중 제약행 지원
            self.Astar = -torch.mm(chunk, A)
            self.Bstar = torch.eye(zdim).to(device) - torch.mm(chunk, B)
            self.bstar = torch.matmul(chunk, b).squeeze(-1)
            self.layers = nn.ModuleList([nn.Linear(idim, hdim)])
            for _ in range(hnum - 1): self.layers.append(nn.Linear(hdim, hdim))
            self.layers.append(nn.Linear(hdim, zdim))
            self.fc_fixed1 = nn.Linear(zdim, zdim, bias=False); self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
            self.fc_fixed2 = nn.Linear(idim, zdim, bias=False); self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)
            self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False)
        def forward(self, x):
            x0 = x
            for l in self.layers[:-1]: x = F.relu(l(x))
            return self.fc_fixed1(self.layers[-1](x)) + self.fc_fixed2(x0)
    return NNOPT


def build_AB(scaler, out_cols, use_energy_hard):
    """A=0(입력 무관), B=질량수지(+선택 에너지수지). 각 행은 outmax 로 스케일(실단위 항등식 보존).
       반환: (A,B,b, brow_energy_scaled)  — brow_energy_scaled 는 soft 페널티에도 사용."""
    zdim = len(out_cols)
    outmax = scaler.scale_[INPUT_DIM:INPUT_DIM+zdim]
    mass_real = build_brow(MASS_BALANCE, out_cols)
    rows = [mass_real * outmax]
    eng_real = build_brow(ENERGY_BALANCE, out_cols) if any(c in out_cols for c in ENERGY_BALANCE) else None
    eng_scaled = (eng_real * outmax) if eng_real is not None else None
    if use_energy_hard and eng_scaled is not None:
        rows.append(eng_scaled)
        print(f"   KKT-hPINN B: 질량수지 + 에너지수지 (2행, hard)")
    else:
        print(f"   KKT-hPINN B: 질량수지 (1행){' + 에너지=soft penalty' if eng_scaled is not None and ENERGY_MODE=='soft' else ''}")
    B = torch.tensor(np.vstack(rows), dtype=torch.float32)
    A = torch.zeros((B.shape[0], INPUT_DIM), dtype=torch.float32)   # A:(m,idim)=0
    b = torch.zeros((B.shape[0], 1), dtype=torch.float32)
    # soft 페널티용 행은 ENERGY_NORM 으로 무차원화(상대잔차). hard 행(rows)은 실단위 유지.
    eng_t = torch.tensor((eng_scaled / ENERGY_NORM).reshape(1,-1), dtype=torch.float32).to(device) if eng_scaled is not None else None
    return A.to(device), B.to(device), b.to(device), eng_t


def energy_residual_report(df_conv, out_cols, scaler):
    """에너지수지 행이 실제 수렴데이터에서 ≈0 으로 닫히는지 진단(실단위). hard 적용 정당성 판단용."""
    if not any(c in out_cols for c in ENERGY_BALANCE):
        print("[energy] 에너지 컬럼 없음 → MB 스키마. (에너지수지 진단 생략)"); return
    eng_real = build_brow(ENERGY_BALANCE, out_cols)
    Z = df_conv[out_cols].values.astype(np.float64)
    resid = Z @ eng_real                       # 실단위 에너지수지 잔차
    typ = np.median(np.abs(Z[:, eng_real != 0]).sum(1)) + 1e-9   # 항들의 전형적 크기
    print(f"[energy] 에너지수지 잔차(실단위): median|r|={np.median(np.abs(resid)):.3g}  "
          f"mean|r|={np.mean(np.abs(resid)):.3g}  max|r|={np.max(np.abs(resid)):.3g}")
    print(f"[energy] 상대잔차 median|r|/Σ|항| ≈ {np.median(np.abs(resid))/typ:.2%}  "
          f"→ {'작음: hard 정당' if np.median(np.abs(resid))/typ < 0.02 else '큼: 항/부호 보정 또는 soft 권장'}")


def train_surrogate(df_conv, out_cols):
    Z0_DIM = len(out_cols)
    arr = df_conv[INPUT_COLS + out_cols].values.astype(np.float64)
    scaler = MaxAbsScaler().fit(arr)
    scl = scaler.transform(arr).astype(np.float32)
    X = torch.tensor(scl[:, :INPUT_DIM]); Y = torch.tensor(scl[:, INPUT_DIM:INPUT_DIM+Z0_DIM])
    n = len(X); nval = int(VAL_RATIO * n)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    vidx, tidx = perm[:nval], perm[nval:]
    tr = DataLoader(TensorDataset(X[tidx], Y[tidx]), batch_size=BATCH, shuffle=True)
    va = DataLoader(TensorDataset(X[vidx], Y[vidx]), batch_size=256, shuffle=False)

    use_hard = (ENERGY_MODE == "hard")
    A, B, b, eng_t = build_AB(scaler, out_cols, use_hard)
    NNOPT = make_NNOPT(Z0_DIM)
    model = NNOPT(INPUT_DIM, HIDDEN_DIM, HIDDEN_NUM, Z0_DIM, A, B, b).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR); lossf = nn.MSELoss()
    use_soft = (ENERGY_MODE == "soft") and (eng_t is not None)
    if use_soft: print(f"[surrogate] 에너지 soft penalty λ={ENERGY_LAMBDA} (실단위 잔차²)")
    best = np.inf; best_sd = None
    print(f"[surrogate] schema={'EB' if Z0_DIM>54 else 'MB'} Z0={Z0_DIM}  train={len(tidx)} val={len(vidx)}  epochs={EPOCHS}  energy={ENERGY_MODE}")
    for ep in range(EPOCHS):
        model.train(); tl = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = lossf(pred, yb)
            if use_soft:   # 스케일된 B행·스케일된 z = 실단위 잔차 → 페널티
                resid = pred @ eng_t.t()
                loss = loss + ENERGY_LAMBDA * torch.mean(resid ** 2)
            loss.backward(); opt.step(); tl += loss.item()
        tl /= len(tr)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device); vl += lossf(model(xb), yb).item()
        vl /= max(len(va), 1)
        if vl < best: best = vl; best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % max(1, EPOCHS // 20) == 0 or ep == 0:
            print(f"  epoch {ep+1:4d}  train_loss {tl:.6f}  val_mse {vl:.6f}  (best {best:.6f})")

    torch.save(best_sd, os.path.join(OUTDIR, "CSDRM_surrogate_sd.pth"))
    np.save(os.path.join(OUTDIR, "surrogate_maxabs.npy"), scaler.max_abs_)
    # ★ downstream(MOO/predict)이 동일 아키텍처를 재구성하도록 메타 저장
    meta = {"schema": "EB" if Z0_DIM > 54 else "MB", "input_dim": INPUT_DIM, "z0_dim": Z0_DIM,
            "output_cols": out_cols, "energy_mode": ENERGY_MODE,
            "mass_balance": MASS_BALANCE, "energy_balance": (ENERGY_BALANCE if Z0_DIM > 54 else {}),
            "B_rows": ["mass"] + (["energy"] if use_hard else [])}
    json.dump(meta, open(os.path.join(OUTDIR, "surrogate_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    model.load_state_dict(best_sd); model.eval()
    with torch.no_grad():
        pv = model(X[vidx].to(device)).cpu().numpy()
    outmax = scaler.max_abs_[INPUT_DIM:INPUT_DIM+Z0_DIM]
    rmse = np.sqrt(np.mean(((pv - scl[vidx.numpy(), INPUT_DIM:]) * outmax) ** 2, axis=0))
    print(f"[surrogate] 저장 완료(meta 포함). val 원단위 RMSE 평균={rmse.mean():.4g}  (best val_mse={best:.6f})")
    return scaler


def train_feasibility(df):
    X = df[INPUT_COLS].values.astype(np.float64)
    y = df['converged'].astype(int).values
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    print(f"\n[feasibility] 전체={len(y)}  성공(pos)={npos}  실패(neg)={nneg}")
    if nneg == 0:
        print("[feasibility] ⚠ 실패 데이터 없음 — 분류기 학습 불가."); return
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    if CLF_KIND == "gb":
        clf = GradientBoostingClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED)
    else:
        clf = RandomForestClassifier(n_estimators=600, class_weight="balanced_subsample",
                                     min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    clf.fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))[:, list(clf.classes_).index(1)]
    pred = (proba >= 0.5).astype(int)
    print("[feasibility] 혼동행렬 (행=실제 0/1, 열=예측 0/1):")
    print(confusion_matrix(yte, pred))
    print(classification_report(yte, pred, target_names=["infeasible(0)", "feasible(1)"], digits=3))
    try: print(f"[feasibility] ROC-AUC = {roc_auc_score(yte, proba):.4f}")
    except Exception: pass
    bundle = {"scaler": sc, "clf": clf, "input_cols": INPUT_COLS, "recommend_thresh": 0.5,
              "note": "MOO 제약: P_feasible(x) >= thresh."}
    joblib.dump(bundle, os.path.join(OUTDIR, "feasibility_clf_bundle.joblib"))
    print(f"[feasibility] 저장 완료: {os.path.join(OUTDIR, 'feasibility_clf_bundle.joblib')}")


def main():
    print(f"라벨 CSV: {MERGED_CSV}  | ENERGY_MODE={ENERGY_MODE}")
    df = pd.read_csv(MERGED_CSV)
    if 'converged' not in df.columns:
        raise ValueError("CSV에 'converged' 컬럼이 없습니다.")
    # ── 스키마 자동 감지: 에너지 컬럼이 다 있고 ENERGY_MODE≠off 이면 EB, 아니면 MB ──
    has_energy = all(c in df.columns for c in ENERGY_COLS)
    use_eb = has_energy and ENERGY_MODE != "off"
    out_cols = OUTPUT_COLS_MB + (ENERGY_COLS if use_eb else [])
    print(f"스키마: {'EB (질량+에너지, ' + ENERGY_MODE + ')' if use_eb else 'MB (질량수지만)'}  | 출력 {len(out_cols)}개"
          + ("" if has_energy else "  (CSV에 에너지 컬럼 없음 → MB 강제)"))
    if MAX_ROWS > 0:
        df = df.sample(min(MAX_ROWS, len(df)), random_state=SEED).reset_index(drop=True)
        print(f"[스모크테스트] {len(df)}행으로 제한")

    if TRAIN_SURROGATE:
        df_conv = df[df['converged'] == 1].dropna(subset=out_cols).reset_index(drop=True)
        print(f"surrogate 학습용 수렴 데이터: {len(df_conv)} / 전체 {len(df)}")
        if use_eb: energy_residual_report(df_conv, out_cols, None)   # hard 정당성 사전 진단
        train_surrogate(df_conv, out_cols)
    else:
        print("surrogate 재학습 건너뜀 (TRAIN_SURROGATE=0).")
    train_feasibility(df)
    print("\n=== 완료 ===")
    print("downstream(MOO/predict)은 surrogate_meta.json 의 z0_dim/output_cols/B_rows 로 동일 아키텍처 재구성할 것.")


if __name__ == "__main__":
    main()
