"""절 AH 작업 목록 (비측정 스윕): 한 설정으로 lego 홀드아웃 20 궤적 + L000 + ficus F_pg_v0 모두 기준 만족 + 서브스텝 로버스트.
줄 형식: <EX_ROOT> <태그> <fem_ex_sim 인자...>. AH_SET 으로 차수, AH_V 로 판 이름 (쉼표) 고르기, AH_N 으로 h_s 배율 (쉼표, 기본 1).
공통 설정 (절 AF) = 절 AD 확정 플래그 + h_s 6.25e-5 s (lego N 267, ficus N 640); 판은 그 위에 한 가지씩 바꾼다."""
import os

W = "/home/dkta/work"
D = os.environ.get("AF_DATA", f"{W}/lbpt/test")
REC = "--mode lag --opt --mass tet --rstat --lim_sr 5 --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 0.5"
HS = 6.25e-5
LEGO = "--floor vel --node_h 0.028284 --frames 80 --no_dump --rbdiag"
FIC = (f"--data {W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp "
       "--no_dump --dump_pred")
TR = [f"{i:04d}" for i in range(20)]


def rep(base, **kw):
    """REC 의 플래그를 바꾼 문자열 (값 None = 지움)."""
    t = base.split()
    o = []
    i = 0
    while i < len(t):
        k = t[i]
        if k.startswith("--") and k[2:] in kw:
            has_v = i + 1 < len(t) and not t[i + 1].startswith("--")
            i += 2 if has_v else 1
            continue
        o.append(k)
        i += 1
    for k, v in kw.items():
        if v is None:
            continue
        o.append(f"--{k}" + ("" if v is True else f" {v}"))
    return " ".join(o)


# 1 차: 원인 확인 -- 하나씩 끄거나 바꾸기 (판 이름 -> (플래그, ficus 도 돌릴지))
V1 = {
    "b0": (REC, False),                                  # 공통 설정 그대로 (강체 지표 기록용)
    "c10": (rep(REC, lim_sr=10), True),                  # 클램프 느슨하게
    "c20": (rep(REC, lim_sr=20), True),
    "keep": (REC + " --lim_v keep", True),               # 클램프가 속도를 줄이지 않음 (위치만)
    "full": (REC + " --vfl_mode full", False),           # 바닥: 접촉 입자 걸린 노드 속도 0 (정답 노드 sticky 꼴)
    "np0": (REC + " --vfl_np 0", False),                 # 바닥: 비관통 끔
    "slip": (REC + " --vfl_mode slip", False),           # 바닥: 미끄럼 (고착 없음, 민감도)
    "vc75": (rep(REC, sp_vcap=0.75), True),              # 셀 탄성 부피 상한
    "vc10": (rep(REC, sp_vcap=1.0), True),               #   상한 없음과 같음
    "h5": (REC + " --vp_h5 H5", False),                  # 정답 float32 위치로 V_p·m_p (버그 수정)
    "nb25": (REC + " --fupd navg --fupd_b 0.25", True),  # 절 AB 매끈한 F 갱신
}
# 2 차: 바닥 정의 정리 (fem_ah_cuda; 0bdde28 의 gtk·rate·gtkr* 태그는 노드 걸음 이중 적분 버그 판 -- 무효, 이름 바꿔 다시) -- 정답 sticky 격자 가중 (gtk) / 단위 시간당 고착 몫 (rate)
V2 = {
    "gk": (REC + " --vfl_w gtk", False),
    "rt": (REC + " --vfl_rate", False),
    "gkr": (REC + " --vfl_w gtk --vfl_rate", False),
    "gkr_np0": (REC + " --vfl_w gtk --vfl_rate --vfl_np 0", False),
    "gk_full": (REC + " --vfl_w gtk --vfl_mode full", False),
}
# 3 차: 비관통 깊이 (정답처럼 얕은 관통 허용) + 조합 ((d) F 갱신은 다른 에이전트 몫이라 뺌)
GKR = REC + " --vfl_w gtk --vfl_rate"
V3 = {
    "gkr_d05": (GKR + " --vfl_npd 0.01", False),
    "gkr_d025": (GKR + " --vfl_npd 0.005", False),
    "gkr_d10": (GKR + " --vfl_npd 0.02", False),
    "rt_d05": (REC + " --vfl_rate --vfl_npd 0.01", False),
    "gkr_vc75": (rep(GKR, sp_vcap=0.75), True),
    "gkr_d05_vc75": (rep(GKR, sp_vcap=0.75) + " --vfl_npd 0.01", False),
    "gkr_h5": (GKR + " --vp_h5 H5", False),
}
# 4 차: 운동량 보존 클램프 (--lim_v mom)
V4 = {
    "mom": (REC + " --lim_v mom", True),
    "gkr_mom": (GKR + " --lim_v mom", False),
    "gkr_d05_mom": (GKR + " --vfl_npd 0.01 --lim_v mom", False),
}
# 5 차: 3 차 상위 조합
V5 = {
    "gkr_d025_vc75": (rep(GKR, sp_vcap=0.75) + " --vfl_npd 0.005", True),
    "gkr_d025_vc6": (rep(GKR, sp_vcap=0.6) + " --vfl_npd 0.005", True),
    "gkr_d05_vc6": (rep(GKR, sp_vcap=0.6) + " --vfl_npd 0.01", True),
}
# 6 차: 비교 -- 절 AF 공통 설정 그대로의 서브스텝 로버스트성 (AH_N 0.5,2)
V6 = {"af": (REC, True)}
# 7 차: 결과 불변 가속 (--vfl_agg) 판 -- 최종 판 (θ 0.75) 과 θ 0.5 로 되돌린 판
V7 = {
    "fa": (rep(GKR, sp_vcap=0.75) + " --vfl_npd 0.005 --vfl_agg", True),
    "fa5": (GKR + " --vfl_npd 0.005 --vfl_agg", True),
}
# 8 차 (조정자 10-31, 정의 변경): 띠 가중 프레임당 한 번 (bf) / 띠 폭 좁게 (bw) / 조합 -- 최종 판 (θ 0.75) 과 θ 0.5 위에서
FINF = rep(GKR, sp_vcap=0.75) + " --vfl_npd 0.005"
FIN5 = GKR + " --vfl_npd 0.005"
V8 = {
    "bf": (FINF + " --vfl_bf", True),
    "bw25": (FINF + " --vfl_band 0.25", False),
    "bw10": (FINF + " --vfl_band 0.1", False),
    "bf_bw25": (FINF + " --vfl_bf --vfl_band 0.25", False),
    "bf5": (FIN5 + " --vfl_bf", True),
    "bw25_5": (FIN5 + " --vfl_band 0.25", False),
}
# 9 차: 띠 더 좁게 -- 0.05 dx_g (1 mm), 0 (평면 아래 입자만, 가중은 정답 sticky 식 그대로)
V9 = {
    "bw05": (FINF + " --vfl_band 0.05", False),
    "bw0": (FINF + " --vfl_band 0", False),
}
# 10 차 (조정자 10-31): 절 AI 클램프 없는 판 (A1+A2+B1, --lim_sr 1e30 = 판정 커널 건너뜀) 위에 띠 0 바닥 -- 늦은 발산이 사라지는지
AIF = " --ai_p2g rs --ai_stress --fupd navg --fupd_b 1"
V10 = {
    "ai_bw0": (rep(FINF + " --vfl_band 0", lim_sr="1e30") + AIF, True),
    "ai_bw0_5": (rep(FIN5 + " --vfl_band 0", lim_sr="1e30") + AIF, True),
}
# 11 차 (절 AH.18): 최종 판 (bwg) + 관성 질량 하한 -- 폭주 노드는 질량이 중앙값의 0.5 % 인 희소 노드
BWG = FINF + " --vfl_band 0 --ah_fast"
V11 = {
    "bwg": (BWG, True),
    "ml02": (BWG + " --ah_mlow 0.02", True),
    "ml05": (BWG + " --ah_mlow 0.05", True),
    "ml10": (BWG + " --ah_mlow 0.1", True),
    "ml25": (BWG + " --ah_mlow 0.25", True),
}
SETS = {"1": V1, "2": V2, "3": V3, "4": V4, "5": V5, "6": V6, "7": V7, "8": V8, "9": V9, "10": V10, "11": V11}
S = SETS[os.environ.get("AH_SET", "1")]
pick = [q for q in os.environ.get("AH_V", "").split(",") if q] or list(S)
NM = [float(q) for q in os.environ.get("AH_N", "1").split(",")]
R = int(os.environ.get("AH_R", "1"))
for r in range(1, R + 1):
    for nm in NM:
        hs = HS * nm
        NL, NF = round((1 / 60) / hs), round(0.04 / hs)
        sfx = "" if nm == 1 else f"_x{nm:g}"
        for v in pick:
            fl, fic = S[v]
            for t in TR + ["L000"]:
                dat = f"{W}/one/L000/L000.pt" if t == "L000" else f"{D}/{t}.pt"
                h5 = f"{W}/lbdata/test_low/L000/sim_0000000000.h5" if t == "L000" else f"{W}/lbdata/test/{t}/sim_0000000000.h5"
                print(f"fem_ah_lego ah_{v}{sfx}_{t}_r{r} --data {dat} --tag T{t} {LEGO} {fl.replace('H5', h5)} --N {NL}")
            if fic:
                print(f"fem_ah_ficus ah_{v}{sfx}_F_r{r} {FIC} {fl} --N {NF}")
