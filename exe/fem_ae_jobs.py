"""절 AE 스윕 (비측정, 카드 1~6): A (매끈한 F 갱신 navg β1) x B (탄성 부피 상한 θ / 메시 칸 V_p) x 클램프 c 3·5.
lego L000 80 프레임 N256 x AE_R 회 (기본 12), ficus F_pg_v0 125 프레임 N640 x 2 회. 줄 형식: <EX_ROOT> <태그> <fem_ex_sim 인자...>"""
import os

LB = "--mode lag --opt --mass tet --no_dump --floor vel --rstat --cfuse --cfuse_mode 2 --N 256"
FD = "/home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt"
FB = (f"--mode lag --opt --mass tet --no_dump --dump_pred --data {FD} --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc "
      "--region_density --pg_vdamp --rstat --cfuse --cfuse_mode 2 --N 640")
NV = "--fupd navg --fupd_b 1"
VC = lambda t: f"--sp vcap --sp_vcap {t}"
VP = "--sp vpmesh"
V = [("ab_c3", f"{NV} --lim_sr 3"),                    # 절 AB 최고 (기준)
     ("ad_vc05_c5", f"{VC(0.5)} --lim_sr 5"),           # 절 AD 최고 (기준)
     ("vpm_c5", f"{VP} --lim_sr 5"), ("vpm_c3", f"{VP} --lim_sr 3"),
     ("nvc05_c3", f"{NV} {VC(0.5)} --lim_sr 3"), ("nvc05_c5", f"{NV} {VC(0.5)} --lim_sr 5"),
     ("nvc1_c3", f"{NV} {VC(1)} --lim_sr 3"), ("nvc1_c5", f"{NV} {VC(1)} --lim_sr 5"),
     ("nvpm_c3", f"{NV} {VP} --lim_sr 3"), ("nvpm_c5", f"{NV} {VP} --lim_sr 5")]
V0 = [("vpm_c0", VP), ("nvpm_c0", f"{NV} {VP}"), ("nvc05_c0", f"{NV} {VC(0.5)}")]   # 클램프 끔 (lego 3 회)
if os.environ.get("AE_SET") == "2":                                                  # 2 차: vpmesh 질량도 같은 정의 (m_p' = ρ V_p') -- 부피만 바꾼 판은 프레임 2 발산
    VM = f"{VP} --sp_vpm_mass"
    V = [("vpmm_c5", f"{VM} --lim_sr 5"), ("vpmm_c3", f"{VM} --lim_sr 3"), ("nvpmm_c3", f"{NV} {VM} --lim_sr 3"), ("nvpmm_c5", f"{NV} {VM} --lim_sr 5")]
    V0 = [("vpmm_c0", VM), ("nvpmm_c0", f"{NV} {VM}")]
if os.environ.get("AE_SET") == "3":                                                  # 3 차: navg β < 1 (ficus 주기를 덜 늘림) x 상한 θ -- 1 차에서 navg1 + θ 는 ficus 주기 2.4~3.2 s (정답 1.4)
    V = [(f"nb{str(b).replace('.', '')}vc{str(t).replace('.', '')}_c{c}", f"--fupd navg --fupd_b {b} {VC(t)} --lim_sr {c}")
         for b, t, c in ((0.5, 0.5, 5), (0.5, 1, 5), (0.25, 0.5, 5), (0.25, 1, 5), (0.5, 0.5, 3), (0.5, 0.75, 5))]
    V0 = []
NR = int(os.environ.get("AE_R", "12"))
J = []
for r in range(1, NR + 1):
    if r <= 2:
        for nm, ar in V:
            J.append(f"fem_ae_ficus ae_f_{nm}_n640_r{r} {FB} {ar}")
    for nm, ar in V:
        J.append(f"fem_ae_lego ae_l_{nm}_n256_r{r} {LB} {ar}")
    if r <= 3:
        for nm, ar in V0:
            J.append(f"fem_ae_lego ae_l_{nm}_n256_r{r} {LB} {ar}")
for j in J:
    print(" ".join(j.split()))
