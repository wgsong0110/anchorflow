#!/bin/bash
# 절 AG.19: 칸 질량 분배 묶음 판 ficus 프레임 3 발산 원인 하나씩 (비측정)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3} AK_SFX=_fdg






bash exe/fem_ak_sweep.sh "v:cmsGm::1::_lr" ficus
bash exe/fem_ak_sweep.sh "v:cmsGm:z:1::_lr" 0000,0016,L000
echo FDG_DONE
