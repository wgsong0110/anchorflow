#!/bin/bash
# 절 AG.18: 서브스텝 국소 CFL 질량 자체가 lego 에서 되는지 (묶음 없이) 대조
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-2} AK_SFX=_dg
bash exe/fem_ak_sweep.sh "v:cmsP:z:1:--ah_mcfl_sub:_sub" 0003,0016
echo DG4_DONE
