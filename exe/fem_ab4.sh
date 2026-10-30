#!/bin/bash
# 절 AB 4 차: 최고 판 (navg β1 + 클램프 c 3, 절 AC 빠른 경로 --cfuse 2) 단독 측정 (GPU0, 실행마다 gpu0.lock) lego N256 x 5·ficus N640 x 3, 같은 세션 기준 (기존 F 갱신 c 3 / c 5) x 2
# -> 최종 F 덤프 (f32) 실행도 단독 측정 + 3DGS 렌더 (카드 1, fem_ab_final.sh). 완주 + 끝 퍼짐 ≤ 1.2 가 아니면 최대 3 번.
# 덧붙여 (비측정, 카드 2): navg1 c3 N256 반복 r5~10 + 이웃 벌어짐 (fspread 40) 1 회.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
OD=/tmp/dkta/lbours
B="--mode lag --opt --mass tet --floor vel --rstat --cfuse --cfuse_mode 2"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --cfuse --cfuse_mode 2"
NV="--fupd navg --fupd_b 1"
( for r in 5 6 7 8 9 10; do EX_ROOT=fem_ab_lego bash exe/fem_ab_run.sh ab2_l_navg1c3_n256_r$r 2 $B --no_dump $NV --lim_sr 3 --N 256 < /dev/null > /dev/null 2>&1; done
  EX_ROOT=fem_ab_lego bash exe/fem_ab_run.sh ab2_l_navg1c3fs_n256_r1 2 $B --no_dump $NV --lim_sr 3 --fspread 40 --N 256 < /dev/null > /dev/null 2>&1
  EX_ROOT=fem_ab_ficus bash exe/fem_ab_run.sh ab2_f_navg1c3fs_n640_r1 2 $FB --no_dump --dump_pred $NV --lim_sr 3 --fspread 62 --N 640 < /dev/null > /dev/null 2>&1
  echo "[비측정 덧붙임 끝] $(date +%T)" ) > $OD/fem_ab_lego/w4.log 2>&1 &
for r in 1 2 3 4 5; do
  EX_TIMED=1 EX_ROOT=fem_ab_lego bash exe/fem_ab_run.sh t4_l_navg1c3_n256_r$r 0 $B --no_dump $NV --lim_sr 3 --N 256 < /dev/null > /dev/null 2>&1
  echo "$(date +%T) t4_l_navg1c3_n256_r$r $(grep -a '롤아웃\]' $OD/fem_ab_lego/t4_l_navg1c3_n256_r$r/v_fem.log | cut -c1-170)"
  if [ $r -le 3 ]; then
    EX_TIMED=1 EX_ROOT=fem_ab_ficus bash exe/fem_ab_run.sh t4_f_navg1c3_n640_r$r 0 $FB --no_dump --dump_pred $NV --lim_sr 3 --N 640 < /dev/null > /dev/null 2>&1
    echo "$(date +%T) t4_f_navg1c3_n640_r$r $(grep -a '롤아웃\]' $OD/fem_ab_ficus/t4_f_navg1c3_n640_r$r/v_fem.log | cut -c1-170)"
  fi
  if [ $r -le 2 ]; then
    EX_TIMED=1 EX_ROOT=fem_ab_lego bash exe/fem_ab_run.sh t4_l_Tc3_n256_r$r 0 $B --no_dump --lim_sr 3 --N 256 < /dev/null > /dev/null 2>&1
    EX_TIMED=1 EX_ROOT=fem_ab_lego bash exe/fem_ab_run.sh t4_l_Tc5_n256_r$r 0 $B --no_dump --lim_sr 5 --N 256 < /dev/null > /dev/null 2>&1
  fi
done
python exe/fem_p_period.py --data $FD --pred $OD/fem_ab_ficus/t4_f_*/pred.pt > $OD/fem_ab_ficus/period4.txt 2>&1
python exe/fem_ab_table.py $OD/fem_ab_lego/t4_l_* > $OD/fem_ab_lego/table4t.md
python exe/fem_ab_table.py $OD/fem_ab_ficus/t4_f_* > $OD/fem_ab_ficus/table4t.md
cat $OD/fem_ab_lego/table4t.md $OD/fem_ab_ficus/table4t.md
rclone copy $OD/fem_ab_lego r2:storage/result/anchorflow/lbours/fem_ab_lego -q; rclone copy $OD/fem_ab_ficus r2:storage/result/anchorflow/lbours/fem_ab_ficus -q
echo "[단독 측정 끝] $(date +%T)"
fin() {  # fin <형상> <태그> <인자...>  -- 완주 + 끝 퍼짐 ≤ 1.2 + 렌더 될 때까지 최대 3 번
  S=$1; TG=$2; shift 2
  for a in 1 2 3; do
    SHAPE=$S EX_ROOT=fem_ab_$S AB_RG=1 bash exe/fem_ab_final.sh ${TG}_F$a "$@" < /dev/null > $OD/fem_ab_$S/${TG}_F$a.final.log 2>&1
    D=$OD/fem_ab_$S/${TG}_F$a
    ok=$(python -c "import json;s=json.load(open('$D/summary.json'));print(int(not s['diverged'] and s['spread_last']<=1.2))" 2>/dev/null)
    [ "$ok" = 1 ] && [ -f $D/vj/RENDER_DONE ] && { echo "[최종] $S $TG: 시도 $a 에서 완주·렌더 $(cat $D/err_fem.txt | tail -1 | cut -c1-160)"; return; }
    echo "[최종] $S $TG: 시도 $a 실패/폭주 시작 (ok=$ok) $(cat $D/err_fem.txt 2>/dev/null | tail -1 | cut -c1-160)"
  done
}
fin lego f_l_navg1c3_n256 $B --dump_f32 $NV --lim_sr 3 --N 256
fin ficus f_f_navg1c3_n640 $FB --dump_f32 --dump_pred $NV --lim_sr 3 --N 640
python exe/fem_p_period.py --data $FD --pred $OD/fem_ab_ficus/f_f_*/pred.pt > $OD/fem_ab_ficus/period4f.txt 2>&1
python exe/fem_ab_table.py $OD/fem_ab_lego/f_l_* > $OD/fem_ab_lego/table4f.md
python exe/fem_ab_table.py $OD/fem_ab_ficus/f_f_* > $OD/fem_ab_ficus/table4f.md
cat $OD/fem_ab_lego/table4f.md $OD/fem_ab_ficus/table4f.md $OD/fem_ab_ficus/period4f.txt
wait
python exe/fem_ab_table.py $OD/fem_ab_lego/ab2_l_* > $OD/fem_ab_lego/table2.md
python exe/fem_ab_table.py $OD/fem_ab_ficus/ab2_f_* > $OD/fem_ab_ficus/table2.md
rclone copy $OD/fem_ab_lego r2:storage/result/anchorflow/lbours/fem_ab_lego -q; rclone copy $OD/fem_ab_ficus r2:storage/result/anchorflow/lbours/fem_ab_ficus -q
echo AB4_DONE $(date +%T)
