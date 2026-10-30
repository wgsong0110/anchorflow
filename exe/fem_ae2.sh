#!/bin/bash
# 절 AE 2 차: 후보 단독 측정 (GPU0, 실행마다 /home/dkta/work/gpu0.lock flock -- fem_ab_run.sh EX_TIMED=1; 시작 전 1~6 번 카드가 비었는지 기록) lego N256 x 5·ficus N640 x 3,
# 같은 세션에서 절 AB 최고·절 AD 최고도 다시 -> 새 최고 판 F 덤프 실행 (단독 측정) + 3DGS 렌더 (카드 1), 완주 + 끝 퍼짐 ≤ 1.2 까지 최대 3 번 -> ficus 진동 곡선 json.
# 후보 목록: exe/fem_ae2_cand.txt (줄 = 이름|lego 인자|ficus 인자), 최종 판 = 첫 줄 아닌 AE_FIN 이름.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
OD=/tmp/dkta/lbours; OL=$OD/fem_ae_lego; OF=$OD/fem_ae_ficus; mkdir -p $OL $OF
RL=r2:storage/result/anchorflow/lbours/fem_ae_lego; RF=r2:storage/result/anchorflow/lbours/fem_ae_ficus
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
LB="--mode lag --opt --mass tet --floor vel --rstat --cfuse --cfuse_mode 2 --N 256"
FB="--mode lag --opt --mass tet --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --cfuse --cfuse_mode 2 --N 640"
( CUDA_VISIBLE_DEVICES=6 python -u exe/fem_ae_vp.py --data $W/one/L000/L000.pt --gt_h5 $W/lbdata/test_low/L000/sim_0000000000.h5 --out $OL/vp_lego2.json > $OL/vp_lego2.log 2>&1
  CUDA_VISIBLE_DEVICES=6 python -u exe/fem_ae_vp.py --data $FD --region_density --gt_h5 $W/lbours/ficus_v0/gt/sim_0000000000.h5 --out $OF/vp_ficus2.json > $OF/vp_ficus2.log 2>&1 )
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee $OL/gpu_before_t.txt
T="env EX_TIMED=1 bash exe/fem_ab_run.sh"
mapfile -t C < exe/fem_ae2_cand.txt
for r in 1 2 3 4 5; do
  for c in "${C[@]}"; do
    IFS='|' read -r nm la fa <<< "$c"
    EX_ROOT=fem_ae_lego $T t_l_${nm}_r$r 0 $LB --no_dump $la < /dev/null > /dev/null 2>&1
    echo "$(date +%T) t_l_${nm}_r$r $(grep -a '롤아웃\]' $OL/t_l_${nm}_r$r/v_fem.log | cut -c1-170)"
    if [ $r -le 3 ]; then
      EX_ROOT=fem_ae_ficus $T t_f_${nm}_r$r 0 $FB --no_dump --dump_pred $fa < /dev/null > /dev/null 2>&1
      echo "$(date +%T) t_f_${nm}_r$r $(grep -a '롤아웃\]' $OF/t_f_${nm}_r$r/v_fem.log | cut -c1-170)"
    fi
  done
done
python exe/fem_p_period.py --data $FD --pred $OF/t_f_*/pred.pt > $OF/period2t.txt 2>&1
python exe/fem_ab_table.py $OL/t_l_* > $OL/table2t.md; python exe/fem_ab_table.py $OF/t_f_* > $OF/table2t.md
for d in $OL/t_l_* $OF/t_f_*; do echo "$(basename $d) 시작 $(tr '\n' ' ' < $d/gpu_at_start.txt) 끝 $(tr '\n' ' ' < $d/gpu_at_end.txt)"; done > $OL/gpu2t.txt
cat $OL/table2t.md $OF/table2t.md $OF/period2t.txt
rclone copy $OL $RL -q; rclone copy $OF $RF -q
echo "[단독 측정 끝] $(date +%T)"
fin() {  # fin <형상> <태그> <인자...>  -- 완주 + 끝 퍼짐 ≤ 1.2 + 렌더 될 때까지 최대 3 번
  S=$1; TG=$2; shift 2
  for a in 1 2 3; do
    SHAPE=$S EX_ROOT=fem_ae_$S AB_RG=1 bash exe/fem_ab_final.sh ${TG}_F$a "$@" < /dev/null > $OD/fem_ae_$S/${TG}_F$a.final.log 2>&1
    D=$OD/fem_ae_$S/${TG}_F$a
    ok=$(python -c "import json;s=json.load(open('$D/summary.json'));print(int(not s['diverged'] and s['spread_last']<=1.2))" 2>/dev/null)
    [ "$ok" = 1 ] && [ -f $D/vj/RENDER_DONE ] && { echo "[최종] $S $TG: 시도 $a 에서 완주·렌더 $(cat $D/err_fem.txt | tail -1 | cut -c1-160)"; return; }
    echo "[최종] $S $TG: 시도 $a 실패/폭주 시작 (ok=$ok) $(cat $D/err_fem.txt 2>/dev/null | tail -1 | cut -c1-160)"
  done
}
c=$(grep "^${AE_FIN:?최종 후보 이름}|" exe/fem_ae2_cand.txt); IFS='|' read -r nm la fa <<< "$c"
fin lego f_l_$nm $LB --dump_f32 $la
fin ficus f_f_$nm $FB --dump_f32 --dump_pred $fa
python exe/fem_p_period.py --data $FD --pred $OF/f_f_*/pred.pt > $OF/period2f.txt 2>&1
python exe/fem_ab_table.py $OL/f_l_* > $OL/table2f.md; python exe/fem_ab_table.py $OF/f_f_* > $OF/table2f.md
cat $OL/table2f.md $OF/table2f.md $OF/period2f.txt
FF=$(ls -d $OF/f_f_${nm}_F* | while read d; do [ -f $d/vj/RENDER_DONE ] && echo $d && break; done)
P="ae=$FF/pred.pt"
for q in "AB=fem_ab_ficus/f_f_navg1c3_n640_F1" "AD=fem_ad_ficus/f_f_vc05_n640_F1"; do
  n=${q%%=*}; p=${q#*=}; mkdir -p $OD/_cv/$n; [ -f $OD/_cv/$n/pred.pt ] || rclone copy r2:storage/result/anchorflow/lbours/$p $OD/_cv/$n -q --include pred.pt
  P="$n=$OD/_cv/$n/pred.pt $P"
done
python exe/fem_ae_curve.py --data $FD --out $OF/curve2.json --pred $P
rclone copy $OL $RL -q; rclone copy $OF $RF -q
rclone check $OF $RF --one-way --size-only -q --include "*/pred.pt" && { rm -f $OF/t_f_*/pred.pt; echo "[정리] R2 확인 뒤 /tmp 단독 측정 ficus pred.pt 지움"; }
echo AE2_DONE $(date +%T)
