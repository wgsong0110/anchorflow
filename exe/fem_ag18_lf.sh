#!/bin/bash
# 절 AG.18: 빈 사면체 포함 칸 묶음 (cmsG) 이 lego 21·ficus 기본 판 대조 (cmsB) 보다 나빠지지 않는지 (비측정 1 회)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3}
L=L000; for k in $(seq 0 19); do L="$L,$(printf %04d $k)"; done
bash exe/fem_ak_sweep.sh "v:cmsG:z:1::_g3" "$L"
bash exe/fem_ak_sweep.sh "v:cmsG::1::_g3" ficus
echo AG18LF_DONE
