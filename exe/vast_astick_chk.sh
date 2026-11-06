#!/bin/bash
# 절 CM17 끄면 같은지 대조 (exe/vast_astick.sh 인스턴스 안, 별 tmux): (A) 판 0000·0013 을 세 가지로 -- 모두 카드 0, 결정적
#   o_dw = 바꾸기 전 커밋 a448383 + astick 과 같은 진단·덤프 플래그 (astick a_<씬> 해시와 같아야 = 새 코드가 끄면 같음)
#   o_pl = a448383 + 절 CM8 a6 명령 그대로 (진단 없음; a6 해시 f444ba6042 · 48a7997a03 와 비교 = a6 이후 커밋들이 (A) 를 바꿨는지)
#   n_pl = 새 커밋 + a6 명령 그대로 (o_pl 과 같아야)
# 출력 /dev/shm/dkta/lbours/fem_cm/astick_chk/<판>_<씬> -> R2 lbours/fem_cm/astick/chk/, 로그 marks/astick_chk.log
NEW=${1:?}; OLD=${2:-a448383}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm/astick_chk; mkdir -p $B
LOG=$B/astick_chk.log; say() { echo "[chk $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $MR/astick_chk.log -q; }
for c in $OLD $NEW; do D=/dev/shm/dkta/impwt/$c; [ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $c && echo $c > $D/.done; }; done
G="--ga hht --rho 0.8 --K 2 --newton 2"; T="--tolN 1e-2 --tolP 1e-1"
DW="--det --v=a --pen --hgt --pdiag --dump --dump_F --dump_part"; PL="--det --v=a"
one() { local nm=$1 c=$2 t=$3; shift 3; local O=$B/${nm}_$t; mkdir -p $O
  ( cd /dev/shm/dkta/impwt/$c && export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_chk_$c && mkdir -p $CUPY_CACHE_DIR && source exe/fem_cm8_scn.sh && cm_scene $t $G --cg 6 $T "$@" \
    && echo "$c $SIM $DA $XA" > $O/run_info.txt && CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY -u $SIM --out $O --tag $t $DA $XA < /dev/null > $O/log.txt 2>&1; echo "rc=$?" >> $O/run_info.txt )
  rm -f $O/pred.pt
  say "$nm $t $(grep -o '"x_sha1": "[0-9a-f]*"\|"diverged": [0-9a-z]*\|"err_mean_done": [0-9.]*' $O/summary.json 2>/dev/null | tr '\n' ' ') $(tail -1 $O/run_info.txt)"
  rclone copy $O $R/lbours/fem_cm/astick/chk/${nm}_$t -q; }
for t in 0000 0013; do one o_dw $OLD $t $DW; one o_pl $OLD $t $PL; one n_pl $NEW $t $PL; done
say "[끝]"
