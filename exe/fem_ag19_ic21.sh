#!/bin/bash
# 절 AG.19: 묶음 판 + IMEX 무리 반복 -- lego 21 (비그래프, 비측정 1 회)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3}
X="--akg_mass_sp@--ah_mcfl_F@--ah_imex@--ah_imex_rel@--ah_imex_cl@1e-3@--no_graph"
L=L000; for k in $(seq 0 19); do L="$L,$(printf %04d $k)"; done
bash exe/fem_ak_sweep.sh "v:cmsGm:z:1:$X:_ic21" "$L"
echo IC21_DONE
