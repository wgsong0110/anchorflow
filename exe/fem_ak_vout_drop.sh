#!/bin/bash
# 절 AK.31 떨어지는 블록만 (fem_ak_vout.sh 의 자료 복원 실패 뒤 다시).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
B=/tmp/dkta/ak_drop2; R=/tmp/dkta/lbours/fem_ak_lego_drop2
AK_KIND=P AK_EN=0 AK_W=1 AK_SFX=_drop2 AK_CARDS='5 6' bash exe/fem_ak_sweep.sh v:nocs,cms:z:1:--vfl_out:_vo drop1 > /dev/null 2>&1
python exe/fem_ak_drop_ana.py $B/init.pt mpm:$B/mpm/x240.pt:0.0041666667 nocs:$R/drop1_nocs_z_P1/pred.pt:0.0166666667 nocs_vo:$R/drop1_nocs_z_vo_P1/pred.pt:0.0166666667 \
  cms:$R/drop1_cms_z_P1/pred.pt:0.0166666667 cms_vo:$R/drop1_cms_z_vo_P1/pred.pt:0.0166666667 nocs_미끄럼:$R/drop1_nocs_z_slip_P1/pred.pt:0.0166666667 2>&1 | grep -v Warning
for d in drop1_nocs_z_vo_P1 drop1_cms_z_vo_P1; do echo "$d $(grep -a '롤아웃\]' $R/$d/v_fem.log | cut -c1-140)"; done
echo "[블록 끝] $(date +%T)"
