#!/bin/bash
# 절 AG.19: 묶음 판 + 입자 F 계수 상계 -- 같은 설정 3 회 반복 판정 (비측정)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-3} AK_SFX=_rep
bash exe/fem_ak_sweep.sh "v:cmsGm:z:3:--akg_mass_sp@--ah_mcfl_F@--ah_mcfl_Fs:_Fs" 0000,0003,0016,0018
bash exe/fem_ak_sweep.sh "v:cmsGm::3:--akg_mass_sp@--ah_mcfl_F@--ah_mcfl_Fs:_Fs" ficus
echo REP_DONE
