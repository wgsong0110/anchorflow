#!/bin/bash
# 절 AG.18: 서브스텝 묶음 CFL 상계 판 (cmsGs) lego 21 (발산 셋 먼저)·ficus (비측정 1 회)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3}
L=0003,0009,0019,L000; for k in 0 1 2 4 5 6 7 8 10 11 12 13 14 15 16 17 18; do L="$L,$(printf %04d $k)"; done
bash exe/fem_ak_sweep.sh "v:cmsGs:z:1::_s5" "$L"
bash exe/fem_ak_sweep.sh "v:cmsGs::1::_s5" ficus
echo LF5_DONE
