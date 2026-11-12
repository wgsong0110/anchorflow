#!/bin/bash
# wolf H9 (= R2 lbours/wolfh12/h8_1 명령) 증분 포텐셜 항별 기록 판 진입점 (코드 디렉토리에서). 사용: bash exe/go_wolfip.sh <커밋> <단계[,단계]>
#   run  = 원래 명령 앞에 exe/wolfip_wrap.py --wip 1 (읽기만, 결과 불변) 을 붙여 한 번 다시 돌림 -> /home/dkta/work/wolfip/run (ip/f*.npz, wolfip.jsonl, pred.pt)
#   an   = 색·합·표본 (CPU) -> /home/dkta/work/wolfip/an
#   ren  = h5 변환 + 렌더 5 개 (일반·합·관성·탄성·장벽+정규화) 를 빈 카드에 병렬로
#   comp = 영상 합성·전체 디코드 확인 + 산점도 HTML -> R2
# 노드 / ·/tmp 안 씀: 코드·TMPDIR·캐시·h5·색 /dev/shm/dkta, 덤프 /home/dkta/work/wolfip. 카드는 공용 표식 (/dev/shm/dkta/anerr_claim) 으로 계산 프로세스 0 인 카드만.
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
R=r2:storage/result/anchorflow/lbours/wolfip
DD=/home/dkta/work/wolfip; O=/dev/shm/dkta/wolfip
mkdir -p $DD $O /dev/shm/dkta/tmp /dev/shm/dkta/cache
export TMPDIR=/dev/shm/dkta/tmp XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wolfip PYTHONUTF8=1 OMP_NUM_THREADS=8 MPLCONFIGDIR=/dev/shm/dkta/cache/mpl_wolfip
PY=/home/dkta/.conda/envs/af/bin/python
L=$DD/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STG; /home $(df -h /home | tail -1 | awk '{print $4}') /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음; 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
GTW=/dev/shm/dkta/ag_data/gt_wolf; WPT=/dev/shm/dkta/ag_data/wolf.pt
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL; NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
claim_wait() { local g; while :; do g=$(claim $1) && { echo $g; return; }; sleep 30; done; }
for S in ${STG//,/ }; do
case $S in
run)
  [ -f $WPT ] && [ -f $GTW/config.json ] || { say "자료 없음 $WPT $GTW"; exit 1; }
  AV=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9); [ "${AV:-0}" -ge 5 ] || { say "/home 여유 ${AV}G < 5G"; exit 1; }
  rm -rf $DD/run; mkdir -p $DD/run
  rclone cat r2:storage/result/anchorflow/lbours/wolfh12/h8_1/cmd.txt | tr -d '\n' | sed -e "s# -u exe/wolfh12_wrap.py # -u exe/wolfip_wrap.py --wip 1 exe/wolfh12_wrap.py #" -e "s# --out [^ ]*\$# --out $DD/run#" -e 's/  */ /g' > $DD/run/cmd.txt
  echo >> $DD/run/cmd.txt; rclone copyto $DD/run/cmd.txt $R/run/cmd.txt -q
  grep -q "wolfip_wrap.py --wip 1 exe/wolfh12_wrap.py" $DD/run/cmd.txt && grep -q -- "--out $DD/run" $DD/run/cmd.txt || { say "명령 바꾸기 실패: $(cut -c1-300 $DD/run/cmd.txt)"; exit 1; }
  G=$(claim_wait $BASHPID); say "run 카드 $G"
  ( while [ ! -f $DD/run/.done ]; do sleep 120; { tail -c 3000 $DD/run/log.txt; echo; echo "ip $(ls $DD/run/ip 2>/dev/null | wc -l) 개"; cat $DD/run/wolfip_err.txt 2>/dev/null | tail -5; } | rclone rcat $R/run/log_tail.txt; done ) &
  env CUDA_VISIBLE_DEVICES=$G timeout 7200 bash -c "$(cat $DD/run/cmd.txt)" < /dev/null > $DD/run/log.txt 2>&1
  rc=$?; rm -f $CL/$G; touch $DD/run/.done
  say "run rc=$rc $(grep -ah '롤아웃\]' $DD/run/log.txt | tail -1 | cut -c1-200) ip $(ls $DD/run/ip 2>/dev/null | wc -l) 개, 오류 $(cat $DD/run/wolfip_err.txt 2>/dev/null | grep -c '^t ') $(grep -a 'rror' $DD/run/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $DD/run $R/run -q --exclude "pred.pt" --transfers 8;;
run2)
  [ -f $WPT ] && [ -f $GTW/config.json ] || { say "자료 없음 $WPT $GTW"; exit 1; }
  AV=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9); [ "${AV:-0}" -ge 5 ] || { say "/home 여유 ${AV}G < 5G"; exit 1; }
  rm -rf $DD/run2; mkdir -p $DD/run2
  rclone cat r2:storage/result/anchorflow/lbours/wolfh12/h8_1/cmd.txt | tr -d '\n' | sed -e "s# -u exe/wolfh12_wrap.py # -u exe/wolfip_wrap.py --wip 1 exe/wolfh12_wrap.py #" -e "s# --out [^ ]*\$# --out $DD/run2#" -e 's/  */ /g' > $DD/run2/cmd.txt
  echo >> $DD/run2/cmd.txt; rclone copyto $DD/run2/cmd.txt $R/run2/cmd.txt -q
  grep -q "wolfip_wrap.py --wip 1 exe/wolfh12_wrap.py" $DD/run2/cmd.txt && grep -q -- "--out $DD/run2" $DD/run2/cmd.txt || { say "명령 바꾸기 실패: $(cut -c1-300 $DD/run2/cmd.txt)"; exit 1; }
  G=$(claim_wait $BASHPID); say "run2 카드 $G"
  ( while [ ! -f $DD/run2/.done ]; do sleep 120; { tail -c 3000 $DD/run2/log.txt; echo; echo "ip $(ls $DD/run2/ip 2>/dev/null | wc -l) 개"; cat $DD/run2/wolfip_err.txt 2>/dev/null | tail -5; } | rclone rcat $R/run2/log_tail.txt; done ) &
  env CUDA_VISIBLE_DEVICES=$G timeout 7200 bash -c "$(cat $DD/run2/cmd.txt)" < /dev/null > $DD/run2/log.txt 2>&1
  rc=$?; rm -f $CL/$G; touch $DD/run2/.done
  say "run2 rc=$rc $(grep -ah '롤아웃\]' $DD/run2/log.txt | tail -1 | cut -c1-200) ip $(ls $DD/run2/ip 2>/dev/null | wc -l) 개, 오류 $(cat $DD/run2/wolfip_err.txt 2>/dev/null | grep -c '^t ') $(grep -a 'rror' $DD/run2/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $DD/run2 $R/run2 -q --exclude "pred.pt" --transfers 8;;
ren2)
  mkdir -p $O/h5p2 $O/vid2; rm -f $O/h5p2/*.h5
  $PY -u exe/lb_roll_to_h5.py --dump $DD/run2/pred.pt --traj $GTW --out $O/h5p2 > $O/h5_2.log 2>&1; say "h5 2 rc=$? $(ls $O/h5p2 | wc -l) 개"
  G=$(claim_wait $BASHPID); say "렌더 2 plain 카드 $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u exe/lb_render.py --config $GTW/config.json --shape wolf --sim $O/h5p2 --frames 50 --out $O/vid2/plain.mp4 < /dev/null > $O/vid2/r_plain.log 2>&1
  rc=$?; rm -f $CL/$G; say "렌더 2 plain rc=$rc $(tail -1 $O/vid2/r_plain.log | cut -c1-200)"; rclone copy $O/vid2 $R/vid2 -q;;
html2)
  $PY -u exe/wolfip_html_color.py $DD/run2 $O/vid2/plain.mp4 $O/wolf_elastic_reg_scatter_color.html $O/tmph2 > $DD/html2.log 2>&1; say "html2 rc=$? $(tail -2 $DD/html2.log | tr '\n' ' ' | cut -c1-400)"
  rclone copyto $DD/html2.log $R/html2.log -q
  [ -s $O/wolf_elastic_reg_scatter_color.html ] && { rclone copyto $O/wolf_elastic_reg_scatter_color.html r2:storage/result/anchorflow/v20261009/wolf_elastic_reg_scatter_color.html -q
    rclone copyto $O/wolf_elastic_reg_scatter_color.html r2:storage/result/anchorflow/lbours/wolfcv/wolf_elastic_reg_scatter_color.html -q; say "R2 올림 2"; };;
an)
  $PY -u exe/wolfip_an.py $DD/run $O/an > $DD/an.log 2>&1; say "an rc=$? $(tail -2 $DD/an.log | cut -c1-600 | tr '\n' ' ')"
  rclone copyto $DD/an.log $R/an.log -q; rclone copy $O/an $R/an -q --include "*.json"
  grep -q WOLFIP_AN_DONE $DD/an.log || exit 1;;
ren)
  mkdir -p $O/h5p $O/vid; rm -f $O/h5p/*.h5
  $PY -u exe/lb_roll_to_h5.py --dump $DD/run/pred.pt --traj $GTW --out $O/h5p > $O/h5.log 2>&1; say "h5 rc=$? $(ls $O/h5p | wc -l) 개"
  rone() { local k=$1 g; g=$(claim_wait $BASHPID); say "렌더 $k 카드 $g"
    local CO=""; [ $k != plain ] && CO="--colors $O/an/col_$k.npy"
    CUDA_VISIBLE_DEVICES=$g $PY -u exe/lb_render.py --config $GTW/config.json --shape wolf --sim $O/h5p --frames 50 $CO --out $O/vid/$k.mp4 < /dev/null > $O/vid/r_$k.log 2>&1
    local rc=$?; rm -f $CL/$g; say "렌더 $k rc=$rc $(tail -1 $O/vid/r_$k.log | cut -c1-200)"; }
  PIDS=""; for k in plain sum in el rb; do rone $k & PIDS="$PIDS $!"; sleep 5; done; wait $PIDS
  rclone copy $O/vid $R/vid -q;;
comp)
  mkdir -p /dev/shm/dkta/wolfip_fonts; rclone copy $R/code/fonts /dev/shm/dkta/wolfip_fonts -q; export WIP_FONT_DIR=/dev/shm/dkta/wolfip_fonts
  $PY -u exe/wolfip_compose.py $O/an/ip.json $O/vid $O/fem_wolf_h9_ip.mp4 $O/tmpc > $DD/compose.log 2>&1; say "compose rc=$? $(tail -2 $DD/compose.log | tr '\n' ' ' | cut -c1-300)"
  $PY -u exe/wolfip_html.py $O/an $O/vid/plain.mp4 $O/wolf_elastic_reg_scatter.html $O/tmph > $DD/html.log 2>&1; say "html rc=$? $(tail -1 $DD/html.log | cut -c1-200)"
  rclone copyto $DD/compose.log $R/compose.log -q; rclone copyto $DD/html.log $R/html.log -q
  [ -s $O/fem_wolf_h9_ip.mp4 ] && { rclone copyto $O/fem_wolf_h9_ip.mp4 r2:storage/result/anchorflow/v20261009/fem_wolf_h9_ip.mp4 -q; rclone copy $O/fem_wolf_h9_ip.mp4 $R -q; }
  [ -s $O/wolf_elastic_reg_scatter.html ] && { rclone copyto $O/wolf_elastic_reg_scatter.html r2:storage/result/anchorflow/v20261009/wolf_elastic_reg_scatter.html -q
    rclone copyto $O/wolf_elastic_reg_scatter.html r2:storage/result/anchorflow/lbours/wolfcv/wolf_elastic_reg_scatter.html -q; }
  say "R2 올림";;
*) say "모르는 단계 $S"; exit 1;;
esac
done
say "끝"; echo "끝 $(date '+%F %T')" > $DD/go_$STG.END; rclone copyto $DD/go_$STG.END $R/go_$STG.END -q
