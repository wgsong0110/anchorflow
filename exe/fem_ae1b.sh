#!/bin/bash
# 절 AE 1b 차: vpmesh 질량도 같은 정의 (--sp_vpm_mass) -- 연기 시험 (lego 5 프레임) 결과를 보고 (프레임 1 KE·제한) 통과면 스윕 (AE_SET=2, AE_R=10, 카드 1~6 x 일꾼 2).
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
OD=/tmp/dkta/lbours; OL=$OD/fem_ae_lego; OF=$OD/fem_ae_ficus; mkdir -p $OL $OF
RL=r2:storage/result/anchorflow/lbours/fem_ae_lego; RF=r2:storage/result/anchorflow/lbours/fem_ae_ficus
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
B="--mode lag --opt --mass tet --no_dump --floor vel --rstat --cfuse --cfuse_mode 2 --N 256 --sp vpmesh --frames 5"
EX_ROOT=fem_ae_lego bash exe/fem_ab_run.sh smoke2_vpm 6 $B --lim_sr 5 < /dev/null > /dev/null 2>&1 &
EX_ROOT=fem_ae_lego bash exe/fem_ab_run.sh smoke2_vpmm 6 $B --sp_vpm_mass --lim_sr 5 < /dev/null > /dev/null 2>&1 &
EX_ROOT=fem_ae_lego bash exe/fem_ab_run.sh smoke2_vpmm_c0 5 $B --sp_vpm_mass < /dev/null > /dev/null 2>&1 &
wait
for t in smoke2_vpm smoke2_vpmm smoke2_vpmm_c0; do echo "== $t"; grep -a "절 AE\|프레임 [0-9]/5\|발산\|롤아웃\]\|Error" $OL/$t/v_fem.log | cut -c1-330; done
python - <<PY || { echo "[연기 시험 실패] 1b 스윕 안 띄움"; exit 1; }
import json; s=json.load(open("$OL/smoke2_vpmm/summary.json")); assert not s["diverged"], s["diverged"]
PY
echo "[연기 시험 통과] 1b 스윕 시작 $(date +%T)"
for G in 1 2 3 4 5 6; do
  for k in 0 1; do
    q=$(( (G - 1) * 2 + k ))
    AE_SET=2 AE_R=10 bash exe/fem_ae_gpu.sh $G $q 12 exe/fem_ae_jobs.py < /dev/null > $OL/w2_${G}_$k.log 2>&1 &
  done
done
wait
echo "[1b 스윕 끝] $(date +%T)"
python exe/fem_ab_table.py $OL/ae_l_* > $OL/table1.md; python exe/fem_ab_table.py $OF/ae_f_* > $OF/table1.md
cat $OL/table1.md $OF/table1.md
rclone copy $OL $RL -q; rclone copy $OF $RF -q
rclone check $OF $RF --one-way --size-only -q --include "*/pred.pt" && { rm -f $OF/ae_f_*/pred.pt; echo "[정리] R2 확인 뒤 /tmp ficus pred.pt 지움"; }
echo AE1B_DONE $(date +%T)
