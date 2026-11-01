#!/bin/bash
# 절 AG.19: 칸 질량 분배 묶음 판 ficus 프레임 3 발산 원인 하나씩 (비측정)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3} AK_SFX=_fdg
bash exe/fem_ak_sweep.sh "v:cmsGm::1:--no_graph@--akpl_cause@--akpl_k@6@--frames@6:_cz" ficus
bash exe/fem_ak_sweep.sh "v:cmsGm::1:--akg_hg@0:_hg0" ficus
bash exe/fem_ak_sweep.sh "v:cmsGm::1:--ah_mcfl@4:_s4" ficus
bash exe/fem_ak_sweep.sh "v:cmsGm::1:--akg_nocpl:_nc" ficus
bash exe/fem_ak_sweep.sh "v:cmsGm::1:--akg_k@100000:_kall" ficus
bash exe/fem_ak_sweep.sh "v:cmsG::1:--akg_mass@--akg_k@0:_k0m" ficus
echo FDG_DONE
