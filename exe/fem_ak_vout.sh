#!/bin/bash
# 절 AK.31 (비측정): 바닥 고착을 접선·안쪽 법선에만 (--vfl_out). 떨어지는 블록 (상한 없는 판·기본 판) -> lego 21 (두 판) -> h_s 0.5·2 배 (두 판). ficus 는 바닥 없음이라 같다.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
B=/tmp/dkta/ak_drop2; SC=L000,$(seq -f %04g 0 19 | paste -sd,)
df -h /tmp | tail -1
for f in drop_f1.pt mpm/x240.pt init.pt; do [ -f $B/$f ] || rclone copy r2:storage/result/anchorflow/lbours/ak_drop2/$(dirname $f) $B/$(dirname $f) -q --include "$(basename $f)"; done
ls -la $B/drop_f1.pt $B/mpm/x240.pt
R=/tmp/dkta/lbours/fem_ak_lego_drop2
AK_KIND=P AK_EN=0 AK_W=1 AK_SFX=_drop2 bash exe/fem_ak_sweep.sh v:nocs,cms:z:1:--vfl_out:_vo drop1 > /dev/null 2>&1
python exe/fem_ak_drop_ana.py $B/init.pt mpm:$B/mpm/x240.pt:0.0041666667 nocs:$R/drop1_nocs_z_P1/pred.pt:0.0166666667 nocs_vo:$R/drop1_nocs_z_vo_P1/pred.pt:0.0166666667 \
  cms:$R/drop1_cms_z_P1/pred.pt:0.0166666667 cms_vo:$R/drop1_cms_z_vo_P1/pred.pt:0.0166666667 nocs_미끄럼:$R/drop1_nocs_z_slip_P1/pred.pt:0.0166666667 2>&1 | grep -v Warning
for d in drop1_nocs_z_vo_P1 drop1_cms_z_vo_P1; do echo "$d $(grep -a '롤아웃\]' $R/$d/v_fem.log | cut -c1-140)"; done
echo "[블록 끝] $(date +%T)"
AK_EN=0 AK_W=1 bash exe/fem_ak_sweep.sh v:nocs,cms:z:1:--vfl_out:_vo $SC > /dev/null 2>&1
AK_EN=0 AK_W=1 bash exe/fem_ak_sweep.sh v:nocs,cms:z:1:--vfl_out@--N@133:_vo_x2 $SC > /dev/null 2>&1 &
AK_EN=0 AK_W=1 bash exe/fem_ak_sweep.sh v:nocs,cms:z:1:--vfl_out@--N@534:_vo_x0.5 $SC > /dev/null 2>&1 &
wait
for v in nocs_z_vo cms_z_vo nocs_z_vo_x2 cms_z_vo_x2 nocs_z_vo_x0.5 cms_z_vo_x0.5; do for t in L000 $(seq -f %04g 0 19); do echo "$t $v $(grep -a '롤아웃\]' /tmp/dkta/lbours/fem_ak_lego/${t}_${v}_r1/v_fem.log | cut -c1-140)"; done; done
echo "[끝] $(date +%T)"
