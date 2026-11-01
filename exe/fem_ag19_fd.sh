#!/bin/bash
# 절 AG.19: 묶음 판 첫 뒤집힘 셀 노드의 실제 접선 강성 대 국소 CFL 상계 (lego 0000·0016, 비그래프, 비측정)
cd "$(dirname "$0")/.." || exit 1
export AK_EN=0 AK_CARDS="${AK_CARDS:-4 5}" AK_W=${AK_W:-2} AK_SFX=_fd
bash exe/fem_ak_sweep.sh "v:cmsGm:z:1:--akg_mass_sp@--no_graph@--akpl_cause@--akpl_k@6@--akg_fd:_fd" 0000,0016
echo FD_DONE
