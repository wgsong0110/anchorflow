"""절 Z 3 차 (비측정 병렬): 로그 장벽 강성·N 보충 -- 2 차에서 κ 1 은 모든 c·N (192~384) 에서 기준보다 일찍 발산.
장벽 강성이 명시적 안정 한계를 줄이는지: κ 0.1·0.01 x c 2·3 x N 192·384·768, κ 1 c 2·3 N 768·1536 (lego 3 회) + 기준 N 768·1536 (3 회)."""
LB = "--mode lag --opt --mass tet --no_dump --floor vel --rstat"
J = []
for r in (1, 2, 3):
    for N in (768, 1536):
        J.append(f"fem_z_lego z_l_ref_n{N}_r{r} {LB} --N {N}")
        for c in ("2", "3"):
            J.append(f"fem_z_lego z_l_sb1c{c}_n{N}_r{r} {LB} --sb 1 --sb_c {c} --N {N}")
    for k, kn in (("0.1", "0p1"), ("0.01", "0p01")):
        for c in ("2", "3"):
            for N in (192, 384, 768):
                J.append(f"fem_z_lego z_l_sb{kn}c{c}_n{N}_r{r} {LB} --sb {k} --sb_c {c} --N {N}")
for j in J:
    print(j)
