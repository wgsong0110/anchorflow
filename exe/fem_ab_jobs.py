"""절 AB 1 차 스윕 (비측정, 카드 1·2): G2P 입자 F 갱신 방식 x 클램프 끔/켬 (σ1/σ3(A_T) ≤ 5, 절 Z 판) -- lego L000 80 프레임 N192 x 3 회, ficus F_pg_v0 125 프레임 N640 x 1 회.
틀은 절 Z 그대로 (--mode lag --opt --mass tet, lego 속도 바닥, ficus 바닥 없음). 줄 형식: <EX_ROOT> <태그> <fem_ex_sim 인자...>"""
import sys
LB = "--mode lag --opt --mass tet --no_dump --floor vel --rstat --fspread 20"
FD = "/home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --mass tet --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --fspread 25"
V = [("T", None), ("bar", "1"), ("bar", "0.5"), ("bar", "0.2"), ("nbar", "1"), ("nbar", "0.5"), ("nbar", "0.2"), ("nbar", "0.05"),
     ("navg", "1"), ("navg", "0.5"), ("polar", "1"), ("polar", "0.5"), ("polar", "0.2")]
CL = [("c0", ""), ("c5", "--lim_sr 5")]
nm = lambda f, b: f + ("" if b is None else b.replace(".", "p"))
fa = lambda f, b: "" if b is None else f"--fupd {f} --fupd_b {b}"
J = []
for r in (1, 2, 3):
    for f, b in V:
        for cn, ca in CL:
            J.append(f"fem_ab_lego ab_l_{nm(f, b)}_{cn}_n192_r{r} {LB} {fa(f, b)} {ca} --N 192")
    if r == 1:                                                                          # ficus: 클램프 켬 (lego 와 같은 플래그) 전부 + 끔은 대표만
        for f, b in V:
            J.append(f"fem_ab_ficus ab_f_{nm(f, b)}_c5_n640_r1 {FB} {fa(f, b)} --lim_sr 5 --N 640")
        for f, b in (("T", None), ("nbar", "1"), ("navg", "1"), ("bar", "1")):
            J.append(f"fem_ab_ficus ab_f_{nm(f, b)}_c0_n640_r1 {FB} {fa(f, b)} --N 640")
for j in J:
    print(" ".join(j.split()))
