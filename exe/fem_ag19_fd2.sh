#!/bin/bash
# 절 AG.19: F 계수 상계 판 0016·0018 반복 3 회 -- 뒤집힘 늘 때마다 실제 강성 대 상계 (비그래프, 비측정)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3} AK_SFX=_fd2
bash exe/fem_ak_sweep.sh "v:cmsGm:z:3:--akg_mass_sp@--ah_mcfl_F@--no_graph@--akpl_cause@--akpl_k@6@--akg_fd_n@40:_f" 0016,0018
echo FD2_DONE
