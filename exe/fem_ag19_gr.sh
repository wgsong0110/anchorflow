#!/bin/bash
# 절 AG.19: 묶음 판 (묶음 입자만 칸 질량) 그래프 대 비그래프 -- lego 0016·0003 (비측정)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3} AK_SFX=_gr
bash exe/fem_ak_sweep.sh "v:cmsGm:z:2:--akg_mass_sp:_g" 0016,0003
bash exe/fem_ak_sweep.sh "v:cmsGm:z:1:--akg_mass_sp@--no_graph:_ng" 0016,0003,0000,0018
echo GR_DONE
