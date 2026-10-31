#!/bin/bash
# 절 AG.18: 탄성 0003 (공중 프레임 3 비유한) 빈 사면체 묶음 판 원인 하나씩 끄기 (비측정)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3} AK_SFX=_dg
S=0003
bash exe/fem_ak_sweep.sh "v:cmsG:z:1::_base2" $S
bash exe/fem_ak_sweep.sh "v:cmsG:z:1::_base2" 0015,L000
bash exe/fem_ak_sweep.sh "v:cmsG::1::_b2" ficus
echo DG_DONE
