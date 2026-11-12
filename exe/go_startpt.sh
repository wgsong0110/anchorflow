#!/bin/bash
# 절 CM113 묶음 startpt 진입점 (exe/go_cellforce.sh 꼴). 코드 디렉토리에서. 사용: bash exe/go_startpt.sh <커밋> <단계[,단계]>
#   chk = 명령줄만, s0 = 연기 (H9 5 프레임 k9, 기록 프레임 3~5), run = H9 wolf (p9_1, wolfh12/h8_1 명령) + H5 wolf (p5_1, wolfbar/b5_1 명령) 각 12 프레임 (기록 f3~f12), an = 집계 (CPU)
#   기록 감싸개 exe/startpt_wrap.py --sp 1 를 가장 바깥에, --dump·--dump_part·--dump_F 만 뺌, --frames 12 (50 프레임 판의 앞 12 프레임과 같은 계산).
# 노드 / ·/tmp 는 안 씀: 코드·TMPDIR·캐시·결과 /dev/shm/dkta.
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
if [[ $STG == *,* ]]; then for s_ in ${STG//,/ }; do bash exe/go_startpt.sh $H $s_ < /dev/null; done; exit 0; fi
R=r2:storage/result/anchorflow/lbours/wolfcv/startpt
RB=r2:storage/result/anchorflow/lbours
O=/dev/shm/dkta/startpt; C=$(pwd -P)
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_startpt
PY=/home/dkta/.conda/envs/af/bin/python
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
RB=r2:storage/result/anchorflow/lbours
say "시작 $(hostname) 커밋 $H 단계 $STG; /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') /home $(df -h /home | tail -1 | awk '{print $4}') 남음; 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
mkcmd() {  # 이름 원본 앞 감싸개 문자열 [프레임]
  local n=$1 s=$2 pre=$3 fr=$4; mkdir -p $O/$n
  rclone cat $RB/$s/cmd.txt | tr -d '\n' | sed -e 's/  */ /g' -e 's/$/ /' -e "s# -u $pre # -u exe/startpt_wrap.py --sp 1 $pre #" -e "s# --out [^ ]* \$# --out $O/$n#" \
    -e 's# --dump_part # #g' -e 's# --dump_F # #g' -e 's# --dump # #g' > $O/$n/cmd.txt
  [ -n "$fr" ] && sed -i "s# --frames [0-9]* # --frames $fr #" $O/$n/cmd.txt
  echo >> $O/$n/cmd.txt
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
}
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL
NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
claim_wait() { local g; while :; do g=$(claim $1) && { echo $g; return; }; sleep 30; done; }
runone() { local n=$1 g; g=$(claim_wait $BASHPID); say "$n 카드 $g 시작"
  rm -rf $O/$n/sp; mkdir -p $O/$n
  env CUDA_VISIBLE_DEVICES=$g timeout 7200 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; rm -f $CL/$g
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) 칸 파일 $(ls $O/$n/sp 2>/dev/null | wc -l) 개 $(du -sh $O/$n/sp 2>/dev/null | cut -f1) 기록 오류 $(python3 -c "import json;d=json.load(open('$O/$n/spinfo.json'));print(d.get('err'), d.get('n_snapdrop'))" 2>/dev/null) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --transfers 8
  say "$n R2 올림 (칸 파일 sp/ 포함, f3~f12 만이라 작음)"; }
runone4() { local n=$1 g; g=$(claim_wait $BASHPID); say "$n 카드 $g 시작"
  rm -rf $O/$n/sp; mkdir -p $O/$n
  env CUDA_VISIBLE_DEVICES=$g timeout 7200 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; rm -f $CL/$g
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) 칸 파일 $(ls $O/$n/sp 2>/dev/null | wc -l) 덤프 $(stat -c %s $O/$n/pred.pt 2>&1)"
  rclone copy $O/$n $R/$n -q --exclude "pred.pt" --transfers 8; say "$n R2 올림 (pred.pt 빼고)"; }
Q=/home/dkta/work/convabl; RQ=r2:storage/result/anchorflow/lbours/wolfcv/convabl
runone5() { local n=$1 g; g=$(claim_wait $BASHPID); say "$n 카드 $g 시작"
  rm -rf $Q/$n/sp; mkdir -p $Q/$n
  env CUDA_VISIBLE_DEVICES=$g timeout 14400 bash -c "$(cat $Q/$n/cmd.txt)" < /dev/null > $Q/$n/log.txt 2>&1
  local rc=$?; rm -f $CL/$g
  say "$n rc=$rc $(grep -ah '롤아웃\]' $Q/$n/log.txt | tail -1 | cut -c1-200) 칸 파일 $(ls $Q/$n/sp 2>/dev/null | wc -l) 덤프 $(stat -c %s $Q/$n/pred.pt 2>&1) $(grep -a 'rror' $Q/$n/log.txt | tail -1 | cut -c1-200)"
  rclone copy $Q/$n $RQ/$n -q --transfers 8; say "$n R2 올림 (덤프 포함, 노드 /home 에 남김)"; }
watch() { local END=$1; shift
  ( while [ ! -f $END ]; do sleep 120; for n in "$@"; do [ -f $O/$n/log.txt ] && { { tail -c 3000 $O/$n/log.txt; echo; echo "칸 파일 $(ls $O/$n/sp 2>/dev/null | wc -l) 개"; tail -c 1500 $O/$n/spinfo.json 2>/dev/null; } | rclone rcat $R/$n/log_tail.txt; }; done; done ) & }
case $STG in
chk) mkcmd p9_1 wolfh12/h8_1 exe/wolfh12_wrap.py 12; mkcmd p5_1 wolfbar/b5_1 'exe/wolfbar_wrap.py --wbdg 1 --wbks 1 exe/speed13_wrap.py' 12; mkcmd k9 wolfh12/h8_1 exe/wolfh12_wrap.py 5
  sed -i 's# -u exe/startpt_wrap.py --sp 1 exe/wolfbar_wrap.py --wbdg 1 --wbks 1 exe/speed13_wrap.py # -u exe/startpt_wrap.py --sp 1 exe/speed13_wrap.py #' $O/p5_1/cmd.txt
  rclone copyto $O/p5_1/cmd.txt $R/p5_1/cmd.txt -q
  for n in k9 p9_1 p5_1; do say "$n: $(cat $O/$n/cmd.txt | cut -c1-6000)"; done;;
s0)  [ -s $O/k9/cmd.txt ] || { say "k9 cmd.txt 없음 -- chk 먼저"; exit 1; }
     rm -f $O/go_s0.RUNEND; watch $O/go_s0.RUNEND k9
     runone k9; touch $O/go_s0.RUNEND
     $PY -u exe/startpt_an.py $O k9 > $O/an_k9.log 2>&1; say "k9 an rc=$? $(tail -4 $O/an_k9.log | tr '\n' ' ' | cut -c1-900)"
     rclone copy $O $R --include "an_k9.log" --include "startpt_k9.json" -q;;
run) for n in p9_1 p5_1; do [ -s $O/$n/cmd.txt ] || { say "$n cmd.txt 없음 -- chk 먼저"; exit 1; }; done
     rm -f $O/go_run.RUNEND; watch $O/go_run.RUNEND p9_1 p5_1
     PIDS=""; for n in p9_1 p5_1; do runone $n & PIDS="$PIDS $!"; sleep 20; done; wait $PIDS; touch $O/go_run.RUNEND;;
an)  $PY -u exe/startpt_an.py $O p9_1 p5_1 > $O/an.log 2>&1; say "an rc=$? $(tail -3 $O/an.log | tr '\n' ' ' | cut -c1-600)"
     rclone copy $O $R --include "an.log" --include "startpt_*.json" -q;;
run2) for n in q9_1 q5_1; do s_=p${n:1}; mkdir -p $O/$n; sed "s# --out [^ ]*\$# --out $O/$n#" $O/$s_/cmd.txt | sed "s#--out $O/$s_ #--out $O/$n #" > $O/$n/cmd.txt; rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q; say "$n: $(cut -c1-300 $O/$n/cmd.txt) ... $(grep -o -- '--out [^ ]*' $O/$n/cmd.txt)"; done
     rm -f $O/go_run2.RUNEND; watch $O/go_run2.RUNEND q9_1 q5_1
     PIDS=""; for n in q9_1 q5_1; do runone $n & PIDS="$PIDS $!"; sleep 20; done; wait $PIDS; touch $O/go_run2.RUNEND;;
run3) for n in r9_1 r5_1; do s_=p${n:1}; mkdir -p $O/$n; sed "s#--out $O/$s_\$#--out $O/$n#" $O/$s_/cmd.txt > $O/$n/cmd.txt; rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q; say "$n: ... $(grep -o -- '--out [^ ]*' $O/$n/cmd.txt)"; done
     rm -f $O/go_run3.RUNEND; watch $O/go_run3.RUNEND r9_1 r5_1
     PIDS=""; for n in r9_1 r5_1; do runone $n & PIDS="$PIDS $!"; sleep 20; done; wait $PIDS; touch $O/go_run3.RUNEND;;
an3) $PY -u exe/startpt3_an.py $O r9_1 r5_1 > $O/an3.log 2>&1; say "an3 rc=$?"
     rclone copy $O $R --include "an3.log" --include "startpt3_*.json" -q;;
run4) for n in w9_1 w0_1; do mkdir -p $O/$n
       sed -e "s#--out $O/p9_1\$#--out $O/$n#" -e "s# --frames 12 # --frames 50 #" -e "s# --bar_cnt # --bar_cnt --dump --dump_part --dump_F #" -e "s#^ *#AF_SP_FR=1,50 #" $O/p9_1/cmd.txt > $O/$n/cmd.txt
     done
     sed -i "s# --w5s 1 # --w5s 0 #" $O/w0_1/cmd.txt
     for n in w9_1 w0_1; do rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q; say "$n: $(grep -o -- '--w5s [0-9]\|--frames [0-9]*\|--dump --dump_part\|--out [^ ]*\|AF_SP_FR=[^ ]*' $O/$n/cmd.txt | tr '\n' ' ')"; done
     rm -f $O/go_run4.RUNEND; watch $O/go_run4.RUNEND w9_1 w0_1
     PIDS=""; for n in w9_1 w0_1; do runone4 $n & PIDS="$PIDS $!"; sleep 20; done; wait $PIDS; touch $O/go_run4.RUNEND;;
vid4) GTW=/dev/shm/dkta/ag_data/gt_wolf
     for n in w9_1 w0_1; do
       ( g=$(claim_wait $BASHPID); V=$O/$n/vj; HD=/dev/shm/dkta/w5s0_vid/$n; mkdir -p $V $HD
         $PY -u exe/lb_roll_to_h5.py --dump $O/$n/pred.pt --traj $GTW --out $HD/h5 > $V/h5.log 2>&1
         CUDA_VISIBLE_DEVICES=$g $PY -u exe/lb_render.py --sim $HD/h5 --config $GTW/config.json --shape wolf --out $V/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $V/r.log 2>&1
         rm -f $CL/$g; rm -rf $HD
         say "$n 영상 $(stat -c %s $V/fem_wolf.mp4 2>&1) $(tail -1 $V/r.log | cut -c1-160)"
         rclone copy $V $R/$n/vj -q ) & sleep 5
     done; wait;;
an4) $PY -u exe/w5s0_an.py $O w9_1 w0_1 > $O/an4.log 2>&1; say "an4 rc=$? $(tail -4 $O/an4.log | tr '\n' ' ' | cut -c1-900)"
     rclone copy $O $R --include "an4.log" --include "w5s0_an.json" -q
     for n in w9_1 w0_1; do rm -f $O/$n/pred.pt; done; say "pred.pt 지움 (노드 /dev/shm 내 덤프, R2 에는 안 올림)";;
cvchk) mkdir -p $Q; H8=$(rclone cat $RB/wolfh12/h8_1/cmd.txt | tr -d '\n' | sed -e 's/  */ /g' -e 's/$/ /')
     B5=$(rclone cat $RB/wolfbar/b5_1/cmd.txt | tr -d '\n' | sed -e 's/  */ /g' -e 's/$/ /' -e 's# -u exe/wolfbar_wrap.py --wbdg 1 --wbks 1 exe/speed13_wrap.py # -u exe/speed13_wrap.py #' -e 's# --bar_cnt # --bar_cnt --dump --dump_part --dump_F #')
     mk() { local n=$1 c="$2"; mkdir -p $Q/$n; echo "$c" | sed -e "s#^ *#AF_SP_FR=1,50 #" -e "s# -u exe/# -u exe/startpt_wrap.py --sp 1 exe/#" -e "s# --out [^ ]* \$# --out $Q/$n#" > $Q/$n/cmd.txt; rclone copyto $Q/$n/cmd.txt $RQ/$n/cmd.txt -q; say "$n: $(cut -c1-8000 $Q/$n/cmd.txt)"; }
     mk a9 "$H8"
     mk b9 "$(echo "$H8" | sed -e 's# --tb_w 2 # --tb_w 2 --tb 1000 #' -e 's# --newton 6 # --newton 60 #' -e 's# --cg 5 # --cg 100 #' -e 's# --tolN 1e-2 # --tolN 1e-4 #' -e 's# --tolP 1e-1 # --tolP 1e-3 #' -e 's# --xit_from 2 # #' -e 's# --xg 1 # #')"
     mk c9 "$(echo "$H8" | sed -e 's# --w8e 1 # --w8e 0 #')"
     mk d5 "$B5";;
cvrun) for n in a9 b9 c9 d5; do [ -s $Q/$n/cmd.txt ] || { say "$n cmd 없음"; exit 1; }; done
     rm -f $Q/go_cvrun.RUNEND
     ( while [ ! -f $Q/go_cvrun.RUNEND ]; do sleep 120; for n in a9 b9 c9 d5; do [ -f $Q/$n/log.txt ] && { tail -c 2500 $Q/$n/log.txt; echo; echo "칸 파일 $(ls $Q/$n/sp 2>/dev/null | wc -l)"; } | rclone rcat $RQ/$n/log_tail.txt; done; done ) &
     PIDS=""; for n in a9 b9 c9 d5; do runone5 $n & PIDS="$PIDS $!"; sleep 20; done; wait $PIDS; touch $Q/go_cvrun.RUNEND;;
cvb) mkdir -p $Q/b9; sed -i 's# --tb_m 0.8 --tb_w 2 # --tb_m 0.8 --tb_w 2 --tb 1000 #' $Q/b9/cmd.txt; rclone copyto $Q/b9/cmd.txt $RQ/b9/cmd.txt -q
     say "b9 (예산 1000 ms = 절 CM93 aC_1 과 같은 꼴; --tc 는 --tb 판이라 --tb 그대로 두고 뒤에 1000): $(grep -o -- '--tb [0-9.]*\|--newton [0-9]*\|--cg [0-9]*\|--tolN [^ ]*\|--tolP [^ ]*\|--xit_from [0-9]*\|--xg [0-9]*' $Q/b9/cmd.txt | tr '\n' ' ')"
     rm -f $Q/go_cvb.RUNEND
     ( while [ ! -f $Q/go_cvb.RUNEND ]; do sleep 120; [ -f $Q/b9/log.txt ] && { tail -c 2500 $Q/b9/log.txt; echo; echo "칸 파일 $(ls $Q/b9/sp 2>/dev/null | wc -l)"; } | rclone rcat $RQ/b9/log_tail.txt; done ) &
     runone5 b9; touch $Q/go_cvb.RUNEND;;
cvvid) GTW=/dev/shm/dkta/ag_data/gt_wolf
     for n in a9 b9 c9; do
       [ -s $Q/$n/vj/fem_wolf.mp4 ] && continue                                    # 이미 렌더한 판은 건너뜀
       [ -s $Q/$n/pred.pt ] || { say "$n 덤프 없음"; continue; }
       ( g=$(claim_wait $BASHPID); V=$Q/$n/vj; HD=/dev/shm/dkta/convabl_vid/$n; mkdir -p $V $HD
         $PY -u exe/lb_roll_to_h5.py --dump $Q/$n/pred.pt --traj $GTW --out $HD/h5 > $V/h5.log 2>&1
         CUDA_VISIBLE_DEVICES=$g $PY -u exe/lb_render.py --sim $HD/h5 --config $GTW/config.json --shape wolf --out $V/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $V/r.log 2>&1
         rm -f $CL/$g; rm -rf $HD
         say "$n 영상 $(stat -c %s $V/fem_wolf.mp4 2>&1) $(tail -1 $V/r.log | cut -c1-160) $(tail -1 $V/h5.log | cut -c1-160)"
         rclone copy $V $RQ/$n/vj -q ) & sleep 5
     done; wait;;
cvan) $PY -u exe/convabl_an.py $Q a9 b9 c9 d5 > $Q/cvan.log 2>&1; say "cvan rc=$? $(tail -5 $Q/cvan.log | tr '\n' ' ' | cut -c1-1500)"
     rclone copy $Q $RQ --include "cvan.log" --include "convabl_an.json" -q;;
an2) $PY -u exe/startpt_an.py $O q9_1 q5_1 > $O/an2a.log 2>&1; say "an2a rc=$?"
     $PY -u exe/startpt2_an.py $O q9_1 q5_1 > $O/an2.log 2>&1; say "an2 rc=$? $(tail -3 $O/an2.log | tr '\n' ' ' | cut -c1-600)"
     rclone copy $O $R --include "an2*.log" --include "startpt_q*.json" --include "startpt2_*.json" -q;;
esac
echo "끝 $(date '+%F %T')" > $O/go_$STG.END; rclone copyto $O/go_$STG.END $R/go_$STG.END -q
say "끝"
