#!/bin/bash
# 절 AG.18: 국소 후퇴 오일러 (--ah_imex) 시험 -- lego 0003·0009·0019 (cmsG 발산), 묶음 판·기본 판 둘 다 (비측정)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3} AK_SFX=_imx
bash exe/fem_ak_sweep.sh "v:cmsGm:z:1::" 0003,0009,0019
echo IMX_DONE
