#!/bin/bash
# 절 AG.18: 묶음 결합 강성 CFL 상계 뒤 cmsG 를 lego 셋·ficus 로 먼저 (비측정)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-5}" AK_W=${AK_W:-2} AK_SFX=_g2
bash exe/fem_ak_sweep.sh "v:cmsG:z:1::" L000,0003,0015
bash exe/fem_ak_sweep.sh "v:cmsG::1::" ficus
echo LF3_DONE
