"""절 R 3 차 (비측정 병렬): 상자 판 이동 제한 하한 Ĵ_min 훑기 -- 2 차에서 0.3 은 lego N 96~256 3/3 완주, 0.1 은 대부분 발산.
lego L000 80 프레임 Ĵ_min 0.2·0.4·0.5 (+ 0.3 N224) x N 160·192·224·256 x 3 회."""
LB = "--mode lag --opt --mass tet --no_dump --floor vel --rstat --lim_mode box"
J = []
for r in (1, 2, 3):
    for jm, nm in ((0.2, "b02"), (0.4, "b04"), (0.5, "b05")):
        for N in (160, 192, 224, 256):
            J.append(f"fem_r_lego r_l_{nm}_n{N}_r{r} {LB} --lim {jm} --N {N}")
    J.append(f"fem_r_lego r_l_b03_n224_r{r} {LB} --lim 0.3 --N 224")
for j in J:
    print(j)
