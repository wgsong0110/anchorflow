#!/bin/bash
# 절 AG.19: 묶음 판 (묶음 입자만 칸 질량) lego 0003·0016 발산 자리 + 모든 입자 Ā 판 lego 21·ficus (비측정)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3}
AK_SFX=_fdg bash exe/fem_ak_sweep.sh "v:cmsGm:z:1:--akg_mass_sp@--no_graph@--akpl_cause@--akpl_k@6:_cz" 0003,0016
L=L000; for k in $(seq 0 19); do L="$L,$(printf %04d $k)"; done
bash exe/fem_ak_sweep.sh "v:cmsGm:z:1:--akg_k@100000:_kall" "$L"
bash exe/fem_ak_sweep.sh "v:cmsGm::1:--akg_k@100000:_kall" ficus
echo DG19_DONE
