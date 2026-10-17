"""Techno-economic (LCOH) and life-cycle (net CO2 emissions) objective functions used for all results."""
import os, math
import numpy as np, pandas as pd

INPUT_DIM = 6
COLS = ["Temperature","Pressure","Oxidant_Methane_Ratio","H2O_CO2_Ratio","Total_Feed_Rate","NG_Feed_Rate"]
SIGN_RULES = {2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:-1,11:1,12:1,13:-1,14:1,15:1,16:1,17:1,18:1,19:1,
              20:1,21:1,22:1,23:1,24:1,25:1,26:1,27:1,28:1,29:1,30:1,31:-1,32:1,33:1,34:1,35:1,36:1,37:1,38:1,39:1,40:1,41:1}

# ── 신규 EB 데이터(P 1–9 bar)용 부호규칙 ──────────────────────────────────────
#  아래 3개는 PINN1(P≤7, MB) 기준으로 만들어진 것이라 새 정의역에서 물리적으로 틀리다.
#   [13] HD_R2      : R-2 가 단열(DUTY=0)이라 원본 데이터에서 48,201행 전부 정확히 0.
#                     정보량이 0 인데 surrogate 는 0 근처 잡음(|중앙값| 8e-05 kW)을 내고,
#                     그 부호 때문에 후보의 약 79% 가 탈락한다. → 규칙 자체가 무의미
#   [ 7] PR_CP2     : 반응압 > 7 bar 면 CP-2 가 팽창기가 되어 일이 음수인 것이 정상.
#   [29] Elec_CP2_kW: 위와 동일. (비용·전력은 calculate_lcoh 에서 0 으로 클램프)
#  → 이 셋을 유지하면 최적화기가 7 bar 위를 아예 탐색하지 못한다 (원본 위반율 25.9%).
_EB_DROP = (7, 13, 29)
SIGN_RULES_EB = {k: v for k, v in SIGN_RULES.items() if k not in _EB_DROP}

# [추가 2026-09-02] PSA 장비비 진단용 배율 (기본 1.0 = 현행 그대로).
PSA_COST_MULT = float(os.environ.get("PSA_COST_MULT", "1.0"))

def setCEPCI():
    return {1988:342.5,1994:368.1,2002:395.6,2006:499.6,2009:521.9,2010:550.8,2011:585.7,2012:584.6,
            2013:567.3,2014:575.7,2015:556.8,2016:541.7,2017:567.5,2018:603.1,2019:607.5,2020:596.2,
            2021:708.8,2022:816.0,2023:800.8}
CV=setCEPCI()
def adj(bc,by,ty=2023): return bc*CV[ty]/CV[by]

def calculate_lcoh(y):
    rF1=adj(3*8000*1/1e6,2002); rCP1=adj(5840*(y[5])**0.82/1e6,1988)
    rR1=adj(math.ceil(y[6]/(6000*0.1*3.14*((0.1/2-0.008)**2)*12))*3500/1e6,2002)
    rF2=adj(3*8000*1/1e6,2002)
    # ── CP-2 처리 (반응압 > 7 bar) ────────────────────────────────────────────
    #  반응압이 분리부 요구압(7 bar)보다 높으면 승압이 불필요해 Aspen 이 CP-2 일을
    #  음수(팽창)로 반환한다. 이때는 압축기가 없는 설계로 간주하여
    #    · 장치비 0        (아래)
    #    · 전력 사용량 0   (TE 에서 y[29] 를 0 으로 클램프)
    #    · 그에 따른 배출 0 (TE 가 calculate_lca 로 전달되므로 자동 반영)
    #  즉 팽창일을 회수 크레딧으로 계상하지 않는다 (보수적).
    rCP2=adj(5840*(y[7])**0.82/1e6,1988) if y[7]>0 else 0.0
    TI=y[8];TO=y[9];HD=y[10]*(-1000)
    rE1=adj(25000*(((HD/(562.5*(((TI-32)-(TO-37))/math.log((TI-32)/(TO-37)))))/9.29)**0.6)/1e6,2002)
    rF3=adj(3*8000*1/1e6,2002)
    # ── PSA 장비비 (2026-08-11 교체) ──────────────────────────────────────────
    #  구식 : adj(1989043*((VF_PSA1/1000)**0.7)/1e6, 2021)   부피유량 m3/hr 멱함수, 2021 기준
    #  신식 : Table S.9 상관식  C = a + b*S^n,  S = PSA 유입 총 몰유량(kmol/hr),
    #         a=19753.8, b=258.9, n=1, cost year 1994  → CEPCI 로 2023 환산
    #  ★ y[11] 의 의미가 VF_PSA1(m3/hr) → MF_PSA1(kmol/hr) 로 바뀌었다.
    #    반드시 2026-08-11 이후 재추출한 데이터(MF_PSA1 열)와 함께 써야 한다.
    #  [추가 2026-09-02] PSA_COST_MULT — 진단용 배율. 기본 1.0(변경 없음).
    #  08-11 상관식 교체로 중앙 운전점 PSA 장비비가 24.40 → 4.16 M$ (5.87배) 떨어졌다.
    #  5.87 을 주면 옛 비용 수준을 흉내내어 "PSA 비용이 프론트 위치를 정하는가" 를 볼 수 있다.
    rPSA1=adj(1989043*((y[11]/1000)**0.7)/1e6,2021)   # R1-C7: PINN1 (pre-08-11) PSA cost, y[11]=VF_PSA1 [m3/hr]
    rHX1=adj(20000*((y[12]/100)**0.6)/1e6,2002)
    # rR2(furnace) 는 -HD_R2 를 쓰는데 Aspen 의 R-2 가 단열이라 HD_R2 는 원본 데이터에서
    # 항상 정확히 0 이다(=> 원본 기준 rR2 도 0). surrogate 는 0 근처 잡음을 내므로
    # y[13]>0 이면 음수의 0.6승이 되어 NaN 이 된다. 원본과 동일한 거동(0)이 되도록 클램프.
    # ※ furnace 장치비가 구조적으로 0 인 것은 별개의 비용모델 이슈로, 여기서는 바꾸지 않는다.
    rR2=adj(7e6*((max(-y[13],0.0))/1e5)**0.6/1e6,2002)
    rHX2=adj(20000*((y[14]/100)**0.6)/1e6,2002); rHX3=adj(20000*((y[15]/100)**0.6)/1e6,2002)
    VF=y[16];H=y[17]
    rP1=adj((2*(math.exp(5.1029-1.2217*math.log(4.403*VF*math.sqrt(0.3048*H))+0.0771*(math.log(4.403*VF*math.sqrt(0.3048*H)))**2))
            *(3*math.exp(8.833-0.6019*math.log(4.403*VF*math.sqrt(0.3048*H))+0.0519*(math.log(4.403*VF*math.sqrt(0.3048*H)))**2)))/1e6,2012)
    TPEC=rF1+rCP1+rR1+rF2+rCP2+rE1+rF3+rPSA1+rHX1+rR2+rHX2+rHX3+rP1
    Deliv=1.1*TPEC; Direct=Deliv*(1+0.47+0.36+0.68+0.11+0.18+0.1+0.7); Indir=Deliv*(0.33+0.41+0.04+0.22+0.44)
    FCI_U=Direct+Indir; r=0.07;n=25; af=(r*(1+r)**n)/((1+r)**n-1); af=math.floor(af*1000)/1000
    AmH2=y[18]*8; AmCO=y[19]*8; AV_CO=AmCO*300/1e6
    AmNG=(y[20]+y[21])*8; AmBFW=y[22]*8; AmCOcap=y[23]*8000; AmCO2cap=y[24]*8; AmCO2=(y[25]-y[26])*8
    TAC=(AmNG*199.12/1e6+AmBFW*0.15/1e6+12.9688*25000/1e6+AmCOcap*7.868*0.5122/1e6+AmCO2cap*40/1e6+AmCO2*70/1e6)
    AOLC=(5*3*37*8*365*(8000/8760)*1.928930279)/1e6
    # y[29]=Elec_CP2_kW. CP-2 가 팽창기로 동작(음수)하면 전력 사용량 0 으로 본다
    # (음의 전력이 크레딧으로 들어가 고압을 인위적으로 유리하게 만드는 것을 차단).
    E_CP2=max(y[29],0.0)
    # [수정 2026-08-11] H2-PSA 전력항 삭제. 구식: + y[28]*12.63*8000  (H2-V 유량 x 12.63 kWh/kg-H2)
    #  → TE 가 줄어들면 유틸리티 비용(Util)과 calculate_lca 의 전력 배출이 함께 감소한다.
    #    (LCA 는 TE 를 인자로 받으므로 별도 수정 없이 자동 반영)
    TE=y[27]*8000+E_CP2*8000+y[28]*12.63*8000+y[30]*8000   # R1-C7: PINN1 keeps the H2-PSA electricity term
    Util=TE*0.115/1e6+(y[31]*0.172)*0.09/1e6
    Maint=0.05*FCI_U; OpSup=0.15*AOLC
    Var=TAC+AOLC+OpSup+Util+Maint+0.15*Maint
    TPC=((TAC+AOLC+OpSup+Util+Maint+0.15*Maint+0.15*AOLC+0.01*FCI_U+0.01*FCI_U
          +0.6*(AOLC+OpSup+Maint)+0.1*(AOLC+OpSup+Maint))/(1-0.05-0.05))
    OpR=np.array([0.5,0.9]+[1]*23)
    tbp=np.sum(AV_CO*OpR); tpcs=np.sum(Var*(1-OpR)-TPC)
    CRF=(0.07*1.07**25)/(1.07**25-1); SU=FCI_U*CRF*24.4*0.1
    AnH2=y[32]*8
    return ((FCI_U*CRF*25-tpcs-tbp+SU)/(AnH2*24.4))*1e3, TE

def calculate_lca(y,TE):
    G={'NG':0.708,'BFW':8.13e-05,'Electricity':0.478,'CO':0.713}   # M3 노트북 기준으로 통일 (구 0.649/0.797)
    H2=y[33]
    return ((y[34]+y[35])/H2*G['NG'] + (y[36])/H2*G['BFW'] + (TE/8000)/H2*G['Electricity']
            - (y[37])/H2*G['CO'] + ((y[38]/44.64)*(64.65+44.57+696.41))/H2*G['Electricity']
            - (y[39]-y[40])/H2 + (y[41])/H2)
