"""절 CM52 묶음 wolfh1 고침 후보 (인자 없으면 이 모듈을 안 부름 = 소스·그래프 그대로).

N (--h1n 1, 정의 변경은 뉴턴 시작점뿐, --y0loc 위): 국소 시작점 섞기 (절 CM27 (c) yl_mix) 의 β_n 을 바닥 법선 성분에만.
  지금: η = (1 − β_n) η(U) + β_n η(ŷ) -- 세 성분 모두 β_n (β_n = 2^-k_n, 바닥 근처 노드 k 25~33 -> 걸음 시작 U 에 거의 그대로).
  N:    Δ = η(ŷ) − η(U), η = η(U) + β_n (Δ·n) n + (Δ − (Δ·n) n) -- 접선 성분은 관성 예측 그대로, 법선 성분만 β_n.
  근거: β 규칙 (rt_bar_kfe 실현 가능 + --y0eb 장벽 에너지 ≤ 걸음 시작) 이 보는 것은 표본점 바닥 높이 d = x_q·n − p_n 뿐이고 (평면 바닥),
  변수 바꿈 y = ŷ + T η 의 T (Tapply, pm 0·1) 는 법선·접선 성분을 섞지 않으므로 y − ŷ 의 법선 성분은 η 의 법선 성분에만 달림 ->
  섞은 뒤 모든 표본점 d·장벽 에너지·yl_chk 판정이 지금과 비트 단위로 같은 식 (같은 β_n), 바뀌는 것은 접선 시작 위치뿐. 수렴하면 같은 답.
  문턱·클램프·노드 바닥 없음. 고정 노드 (pm 2) 는 그대로 (지금처럼 건너뜀).
"""
ST = dict(n=0)

OLD = "    float e[3]; for (int a=0;a<3;a++) { e[a] = (1.f - be)*eU[n*3+a] + be*eS[n*3+a]; eta[n*3+a] = e[a]; }"
NEW = ("    float dl_[3]; for (int a=0;a<3;a++) dl_[a] = eS[n*3+a] - eU[n*3+a];                       // 절 CM52 N: β 를 법선 성분에만\n"
       "    float dn_ = dl_[0]*n0 + dl_[1]*n1 + dl_[2]*n2; float nn_[3] = {n0, n1, n2};\n"
       "    float e[3]; for (int a=0;a<3;a++) { e[a] = eU[n*3+a] + be*dn_*nn_[a] + (dl_[a] - dn_*nn_[a]); eta[n*3+a] = e[a]; }")


def install(n=0):
    from . import fem_an_rt as ART
    ST["n"] = int(n)
    if ST["n"] == 1:
        assert ART.Y0L_SRC.count(OLD) == 1, "절 CM52 N: yl_mix 섞기 줄을 못 찾음"
        ART.Y0L_SRC = ART.Y0L_SRC.replace(OLD, NEW)
