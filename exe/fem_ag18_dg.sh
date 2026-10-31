#!/bin/bash
# 절 AG.18: 탄성 0003 (공중 프레임 3 비유한) 빈 사면체 묶음 판 원인 하나씩 끄기 (비측정)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3} AK_SFX=_dg
S=0003
bash exe/fem_ak_sweep.sh "v:cmsG:z:1::_base" $S
bash exe/fem_ak_sweep.sh "v:cmsG:z:1:--akg_hg@0:_hg0" $S
bash exe/fem_ak_sweep.sh "v:cmsG:z:1:--akg_m0fix:_m0" $S
bash exe/fem_ak_sweep.sh "v:cmsG:z:1:--akg_nocpl:_nc" $S
bash exe/fem_ak_sweep.sh "v:cmsG:z:1:--ah_mcfl@4:_s4" $S
bash exe/fem_ak_sweep.sh "v:cmsP:z:1:--akg_k@1@--akg_hg@1:_noemp" $S
bash exe/fem_ak_sweep.sh "v:cmsG:z:1:--no_graph:_ng" $S
bash exe/fem_ak_sweep.sh "v:cmsG::1:--floor@none:_fl0" $S
echo DG_DONE
