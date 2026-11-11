#!/bin/bash
# 절 CM106.7 단계 en: TC ficus 판 vCF 덤프 (pred.pt: 위치 + 입자 F, 같은 명령 --dump --dump_F --dump_part) 로 입자별 ψ(F_T) 색 3DGS 렌더.
# stab_tetc.sh 가 'en' 단계에서 부름 (사용: bash exe/tetc_en.sh <R2 tetc 경로>). 노드 / ·/tmp 안 씀: 산출물 /home/dkta/work/tetc_energy, 임시 /dev/shm/dkta.
export PATH=/home/dkta/bin:$PATH
R=$1; PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR
OD=/home/dkta/work/tetc_energy; HD=/dev/shm/dkta/tetc_en; FGT=/home/dkta/work/lbours/ficus_v0/gt
mkdir -p $OD $HD
LG=/dev/shm/dkta/lbours/tetc/tetc_en.log; mkdir -p $(dirname $LG)
log() { echo "[$(date '+%F %T')] $*" >> $LG; rclone copyto $LG $R/en/tetc_en.log -q; }
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL; NG=$(nvidia-smi -L | wc -l)
claim() { local g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $$ > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
log "시작 $(hostname) 코드 $(pwd -P)"
if [ ! -s $OD/pred.pt ]; then
  [ -s /home/dkta/work/tetc_dump/vCF/pred.pt ] && cp /home/dkta/work/tetc_dump/vCF/pred.pt $OD/pred.pt || rclone copyto $R/vCF/pred.pt $OD/pred.pt -q
fi
rclone copyto $R/vCF/cmd.txt $OD/cmd_vCF.txt -q
log "pred.pt $(stat -c %s $OD/pred.pt)"
$PY -u exe/tetc_en_psi.py $OD/pred.pt $FGT/config.json $OD > $OD/psi.log 2>&1; log "psi rc=$? $(tail -2 $OD/psi.log | cut -c1-300)"
[ -s $OD/colors.npy ] || { log "colors 없음 -- 중단"; exit 1; }
$PY -u exe/lb_roll_to_h5.py --dump $OD/pred.pt --traj $FGT --out $HD/h5 > $OD/h5.log 2>&1; log "h5 rc=$? $(tail -1 $OD/h5.log | cut -c1-200)"
G=""; for i in $(seq 1 60); do G=$(claim) && break; G=""; sleep 30; done
[ -n "$G" ] || { log "빈 카드 없음 -- 중단"; exit 1; }
log "카드 $G"
CUDA_VISIBLE_DEVICES=$G $PY -u exe/lb_render.py --sim $HD/h5 --config $FGT/config.json --shape ficus --out $OD/energy.mp4 --colors $OD/colors.npy --frame_label 0 --caption " " < /dev/null > $OD/render.log 2>&1
log "render rc=$? $(stat -c %s $OD/energy.mp4 2>&1) $(tail -1 $OD/render.log | cut -c1-200)"
rm -f $CL/$G; rm -rf $HD
rclone copy $OD $R/en -q --exclude pred.pt --exclude colors.npy; log "R2 올림 끝 (en/)"
