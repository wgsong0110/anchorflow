#!/bin/bash
# 절 CM106.8 단계 inv (사용자 11-11): 레고 TC 진단 판 dL_1 과 같은 명령 + 뒤집힌 사면체 전부 기록 (exe/tetc_inv_wrap.py) + 영상용 덤프 (--dump --dump_F --dump_part), 판 N 개 (기본 2, 카드 하나씩).
# stab_tetc.sh 가 'inv' 단계에서 부름 (사용: bash exe/tetc_inv.sh <R2 tetc 경로>). 노드 / ·/tmp 안 씀: 산출물 /home/dkta/work/tetc_inv, 임시 /dev/shm/dkta.
export PATH=/home/dkta/bin:$PATH
R=$1; NP=${2:-2}; PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR
OD=/home/dkta/work/tetc_inv; HD=/dev/shm/dkta/tetc_inv; LGT=/home/dkta/work/lbours/../lbdata
mkdir -p $OD $HD /dev/shm/dkta/anerr_data
GTD=/home/dkta/work/lbdata/test_low/L000; CFG=/home/dkta/work/lbdata/config.json
LG=/dev/shm/dkta/lbours/tetc/tetc_inv.log; mkdir -p $(dirname $LG)
log() { echo "[$(date '+%F %T')] $*" >> $LG; rclone copyto $LG $R/inv/tetc_inv.log -q; }
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL; NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
log "시작 $(hostname) 코드 $(pwd -P) 판 $NP; / $(df -h / | tail -1 | awk '{print $4}') /home $(df -h /home | tail -1 | awk '{print $4}') shm $(df -h /dev/shm | tail -1 | awk '{print $4}') 남음"
[ -s /dev/shm/dkta/anerr_data/L000.pt ] || cp /home/dkta/work/one/L000/L000.pt /dev/shm/dkta/anerr_data/L000.pt
rclone copyto $R/dL_1/cmd.txt $OD/cmd_dL_1.txt -q
[ -s $OD/cmd_dL_1.txt ] || { log "dL_1 cmd.txt 없음 -- 중단"; exit 1; }
runone() { local n=$1 me=$BASHPID G=""
  for i in $(seq 1 240); do G=$(claim $me) && break; G=""; sleep 30; done
  [ -n "$G" ] || { log "$n 빈 카드 없음 -- 중단"; return 1; }
  log "$n 카드 $G"
  local d=$OD/$n; rm -rf $d; mkdir -p $d
  # dL_1 명령 그대로: 감싸개만 tetc_inv_wrap 으로, --out 바꾸고, 덤프 세 플래그 추가 (--K 1 뒤)
  sed -e "s#exe/tetc_wrap.py#exe/tetc_inv_wrap.py#" -e "s# --K 1 # --K 1 --dump --dump_F --dump_part #" -e "s#--out /home/dkta/work/tetc_dump/dL_1#--out $d#" $OD/cmd_dL_1.txt > $d/cmd.txt
  grep -q "tetc_inv_wrap.py" $d/cmd.txt && grep -q -- "--dump_part" $d/cmd.txt && grep -q -- "--out $d\$" $d/cmd.txt || { log "$n 명령 고치기 실패"; rm -f $CL/$G; return 1; }
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $d/apps_at_start.txt 2>&1
  log "$n 시작 (카드 $G)"
  env CUDA_VISIBLE_DEVICES=$G timeout 5400 bash -c "$(cat $d/cmd.txt)" < /dev/null > $d/log.txt 2>&1
  log "$n 계산 rc=$? $(grep -a '롤아웃\|발산' $d/log.txt | tail -2 | cut -c1-200 | tr '\n' ' ')"
  if [ -s $d/pred.pt ]; then
    $PY -u exe/lb_roll_to_h5.py --dump $d/pred.pt --traj $GTD --out $HD/$n/h5 > $d/h5.log 2>&1; log "$n h5 rc=$? $(tail -1 $d/h5.log | cut -c1-200)"
    CUDA_VISIBLE_DEVICES=$G $PY -u exe/tetc_inv_render.py --sim $HD/$n/h5 --config $CFG --shape lego --inv $d --out $d/png < /dev/null > $d/render.log 2>&1; log "$n 그림 rc=$? $(tail -1 $d/render.log | cut -c1-200)"
    tar -C $d -cf $d/png.tar png && rm -rf $d/png
  else log "$n pred.pt 없음"; fi
  rm -rf $HD/$n; [ "$(cat $CL/$G 2>/dev/null)" = "$me" ] && rm -f $CL/$G
  rclone copy $d $R/inv/$n -q --transfers 8; log "$n R2 올림 끝 (inv/$n)"
}
for k in $(seq 1 $NP); do runone iL_$k & sleep 20; done
wait
log "끝"
