#!/bin/bash
# [blx] Spring-Gaus bread 덤프·물리·시각 지표를 다른 카드에서 병렬로 (조정자 11-17: 12 시 전 완료; 시간 1 회는 학습 카드 final_blx_sgf).
#   지금: 코드·준비 판 정적 파일·정답 렌더 PNG·pyiqa/RAFT 가중치 미리 (정답 PNG 와 가중치 데움은 GPU 가볍게). 학습 끝 (R2 SGB/MV_TRAIN_DONE) -> sg_train.json·마지막 exp 받아
#   pr2_grid BLX_NOTIME=1 BLX_DUMP=1 (OUT 은 이 인스턴스 다른 일과 안 겹치게 /workspace/blxd, 준비 판은 sg_train.json 절대 경로 자리) -> R2 SGB/g 로, 표식 marks_SGB/DUMP_DONE.
#   GPU 를 쓰는 일 (정답 렌더·가중치 데움·덤프) 은 BLX_WAIT_SESS (기본 ipgwt25r: i-PG 단독 시간 측정) tmux 세션이 끝난 뒤에만.
#   사용: bash blx_dump_bread.sh <커밋>
set -u
H=${1:?커밋}; export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1
RA=r2:storage/result/anchorflow; OX=$RA/lbours/final_pa/blx; T=SGB
P=/dev/shm/dkta/lbours/paper_remeasure2/stab/p; F=$P/springgaus_bread_prep; O=/workspace/blxd; L=/root/blxdump.log
log() { echo "[blxdump $(date -u '+%F %T')] $*" | tee -a $L; rclone copyto $L $OX/marks_$T/blxdump.log -q; }
D=/root/blx_code_$H; mkdir -p $D $O/p $P; rclone cat $OX/code/blx_$H.tar | tar -x -C $D
[ -e $O/p/springgaus_bread_prep ] || ln -s $F $O/p/springgaus_bread_prep
mkdir -p /tmp/dkta 2>/dev/null; [ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta; mkdir -p /tmp/dkta/ag_data
[ -e /tmp/dkta/ag_data/gt_bread ] || ln -s /dev/shm/dkta/ag_data/gt_bread_full /tmp/dkta/ag_data/gt_bread
export ST_SG_PYPATH=/dev/shm/dkta/pylib/sg ST_SG_REPO=/dev/shm/dkta/ext/Spring-Gaus PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -s $F/bread.yaml ] || (cd $P && rclone cat $OX/code/sg_fit_bread.tar | tar -x)
rclone copy $OX/$T/p/springgaus_bread_prep $F -q --max-depth 1 --exclude "log_*"
log "준비: 코드 $H, 준비 판 $(ls $F | tr '\n' ' ')"
cd $D
log "GPU 일 대기: 이 카드의 다른 에이전트 단독 시간 측정 세션 (${BLX_WAIT_SESS:-ipgwt25r}) 끝날 때까지 (조정자 11-17)"
for ss in ${BLX_WAIT_SESS:-ipgwt25r}; do while tmux has-session -t $ss 2>/dev/null; do sleep 30; done; done
G=$O/gt_bread; mkdir -p $G
if [ ! -f $G/done ]; then
  CUDA_VISIBLE_DEVICES=0 python -u exe/lb_render.py --sim /tmp/dkta/ag_data/gt_bread --config /tmp/dkta/ag_data/gt_bread/config.json --shape bread --out $G/gt_bread.mp4 --png $G/png --frames 300 < /dev/null > $G/r.log 2>&1
  [ -f $G/png/0001.png ] && echo ok > $G/done
fi
log "정답 PNG $(ls $G/png 2>/dev/null | wc -l) 장"
CUDA_VISIBLE_DEVICES=0 python -u exe/ffx_vis.py --png $G/png --out /root/blx_warm_vis.json > /root/blx_warm_vis.log 2>&1
log "pyiqa·RAFT 데움: $(tail -1 /root/blx_warm_vis.log | cut -c1-120)"
log "대기: R2 $T/MV_TRAIN_DONE"
until [ -n "$(rclone lsf $OX/$T/MV_TRAIN_DONE 2>/dev/null)" ]; do sleep 30; done
rclone copy $OX/$T/p/springgaus_bread_prep $F -q --max-depth 1 --exclude "log_*"
EL=$(python -c "import json;print(json.load(open('$F/sg_train.json'))['exp'])")
rclone copy $OX/$T/p/springgaus_bread_prep/exp/$(basename $EL) $EL -q --transfers 8 --exclude "evaluations/**" --exclude "dynamic_refine/**"
python -c "import json;n=json.load(open('$F/sg_train.json'))['n_step'];json.dump(dict(rc=0,hi=int(n),card='0',resume='sg_resume',host='dump'),open('$F/res.json','w'),indent=1)"
log "학습 결과 받음: $(cat $F/res.json | tr -d '\n ') exp $EL ($(ls $EL/checkpoints_dynamic/checkpoint 2>/dev/null | tr '\n' ' '))"
BLX_NOTIME=1 BLX_DUMP=1 python -u exe/pr2_grid.py --out $O --jobs exe/blx_jobs_sgb.json --reps 1 --scenes L000 \
  --until $(( $(date +%s) + 20 * 3600 )) --hard $(( $(date +%s) + 21 * 3600 )) >> $O/grid_main.log 2>&1
log "[덤프·지표 끝] $(grep -h '덤프' $O/stab.log | tail -1 | cut -c1-300)"
rclone copy $O/g $OX/$T/g -q --transfers 8 --exclude "*/h5/**"; rclone copy $O/gt_bread $OX/$T/gt_bread -q --transfers 8
rclone copyto $O/stab.log $OX/$T/stab_dump.log -q
rclone check $O/g $OX/$T/g --one-way --size-only -q --exclude "*/h5/**" && { echo "$(date)" | rclone rcat $OX/marks_$T/DUMP_DONE; log "[R2 확인 차이 0] DUMP_DONE"; } || log "[R2 확인 차이 있음]"
