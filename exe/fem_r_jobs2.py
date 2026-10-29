"""절 R 2 차 스윕: (2) 이동 제한 상자 판 (--lim_mode box, 서브스텝당 커널 2 개) -- 1 차의 8 색 판 (gs) 이 커널 8 개/서브스텝으로 느려서 (lego N96 7 -> 27 ms, ficus N640 13 -> 100 ms, 비측정 병렬).
  b01 / b03 = det A_T ≥ 0.1 / 0.3 (속도도 α 배), b01k = 속도 그대로, fX1b01 / fA1b01 = (1)+(2)
lego L000 80 프레임 N 64~384 x 3 회, ficus 125 프레임 N 384·448·512·640·768 x 1 회 (+ 기준 384·448)."""
W = "/home/dkta/work"
V = {"b01": "--lim 0.1 --lim_mode box", "b03": "--lim 0.3 --lim_mode box", "b01k": "--lim 0.1 --lim_mode box --lim_v keep",
     "fX1b01": "--fb 1 --lim 0.1 --lim_mode box", "fA1b01": "--fb 1 --fb_on A --lim 0.1 --lim_mode box"}
LB = "--mode lag --opt --mass tet --no_dump --floor vel --rstat"
FD = f"{W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = f"--mode lag --opt --mass tet --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat"
J = []
for r in (1, 2, 3):
    for v in V:
        for N in (64, 96, 128, 160, 192, 256, 384):
            J.append(f"fem_r_lego r_l_{v}_n{N}_r{r} {LB} {V[v]} --N {N}")
for v in ("b01", "b03", "fX1b01", "fA1b01"):
    for N in (384, 448, 512, 640, 768):
        J.append(f"fem_r_ficus r_f_{v}_n{N} {FB} {V[v]} --N {N}")
for v, a in (("base", ""), ("fX1", "--fb 1"), ("fA1", "--fb 1 --fb_on A")):
    for N in (384, 448):
        J.append(f"fem_r_ficus r_f_{v}_n{N} {FB} {a} --N {N}")
for j in J:
    print(j)
