#!/bin/bash
# 절 AG.18 공통 설정 (비측정 1 회): lego 21 (띠 0 바닥) + ficus -- 기본 판 + 프레임 단위 국소 CFL 질량
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3}
L=L000; for k in $(seq 0 19); do L="$L,$(printf %04d $k)"; done
bash exe/fem_ak_sweep.sh "v:cmsU:z:1::_u" "$L"
bash exe/fem_ak_sweep.sh "v:cmsU::1::_u" ficus
echo UNI_DONE
