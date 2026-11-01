#!/bin/bash
# 절 AG.19: 묶음 판 (묶음 입자 칸 질량, 프레임 시작 F 상계) + IMEX 무리 반복 (상대 꼴) -- 3 회 반복 (비그래프, 비측정)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3} AK_SFX=_ic
X="--akg_mass_sp@--ah_mcfl_F@--ah_imex@--ah_imex_rel@--ah_imex_cl@1e-3@--no_graph"
bash exe/fem_ak_sweep.sh "v:cmsGm:z:3:$X:_ic" 0000,0003,0016,0018
bash exe/fem_ak_sweep.sh "v:cmsGm::3:$X:_ic" ficus
echo IC_DONE
