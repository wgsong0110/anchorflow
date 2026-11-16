#!/bin/bash
# [final_fix_mpmbl] 인스턴스 재시작 뒤 /dev/shm 자료 복구 (지워진 것만) 후 i-PG 판 재개.  사용: bash exe/mpmbl_restore.sh <작업> ...
#   작업: wolf_fr:<N> (오차 판 이어), wolf_t:<N> (단독 시간 판 처음부터: 기존 로컬·R2 체크포인트 이름 바꿔 둠), lsrej:<장면>:<판> (이어)
set -u
R=r2:storage/result/anchorflow; MI=$R/_vastmirror; RB=$R/lbours/final_fix/bl_mpm; W=/home/dkta/work; SD=/dev/shm/dkta/anerr_data; AG=/dev/shm/dkta/ag_data; NW=$W/newsc
export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1
mkdir -p $SD $AG /dev/shm/dkta/cache; cd /root/mb/code
need_wolf() { [ -s $AG/wolf.pt ] || rclone copyto $R/lbours/fem_ag_data/wolf.pt $AG/wolf.pt -q
  [ -s $AG/gt_wolf/config.json ] || python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/mb/gtwolf_unused.pt --out_dir $AG/gt_wolf > /root/mb/gt_wolf.log 2>&1; }
for job in "$@"; do
  IFS=: read -r kind a b <<< "$job"
  case $kind in
    wolf_fr) need_wolf; tmux new-session -d -s ipgw${a}r "export AF_IPG_FAST=1 AF_LS_REJECT=1; until bash /root/mb/code/exe/mpmbl_ipgwolf_full.sh fr $a >> /root/mb/ipgw${a}r.out 2>&1; do echo retry >> /root/mb/ipgw${a}r.out; sleep 60; done";;
    wolf_t) need_wolf; O=/workspace/mb/ipg_wolf; T=N${a}e5_frt; ts=$(date +%H%M)
            [ -e $O/$T ] && mv $O/$T $O/${T}_stopped$ts; [ -e $O/$T.log ] && mv $O/$T.log $O/${T}_stopped$ts.log
            rclone lsf $RB/ipg_wolf/$T/ckpt.pt >/dev/null 2>&1 && rclone moveto $RB/ipg_wolf/$T/ckpt.pt $RB/ipg_wolf/${T}_stopped$ts/ckpt.pt -q
            rclone lsf $RB/ipg_wolf/$T.log >/dev/null 2>&1 && rclone moveto $RB/ipg_wolf/$T.log $RB/ipg_wolf/${T}_stopped$ts.log -q
            tmux new-session -d -s ipgwt${a}r "export AF_LS_REJECT=1; bash /root/mb/code/exe/mpmbl_ipgwolf_timewait.sh frt $a > /root/mb/ipgwt${a}r.out 2>&1";;
    lsrej) case $a in
             L000) [ -s $SD/L000.pt ] || rclone copyto $R/vastpkg/L000.pt $SD/L000.pt -q;;
             bread) [ -s $AG/bread.pt ] || rclone copyto $R/lbours/fem_ag_data/bread.pt $AG/bread.pt -q
                    [ -s $AG/gt_bread_full/config.json ] || python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/mb/gtbread_unused.pt --out_dir $AG/gt_bread_full > /root/mb/gt_bread.log 2>&1;;
             pillow2sofa) if [ -L $NW/pillow2sofa_h5 ]; then mkdir -p /dev/shm/p2s/pillow2sofa_h5; rclone copy $R/lbours/paper_remeasure2/scenes/pillow2sofa/h5 /dev/shm/p2s/pillow2sofa_h5 -q --transfers 16; fi;;
           esac
           tmux new-session -d -s lsrej "until bash /root/mb/code/exe/mpmbl_lsrej.sh $a $b >> /root/mb/lsrej.out 2>&1; do echo retry >> /root/mb/lsrej.out; sleep 60; done"
           sleep 2; tmux has-session -t lspost 2>/dev/null || tmux new-session -d -s lspost "while tmux has-session -t lsrej 2>/dev/null; do sleep 30; done; bash /root/mb/code/exe/mpmbl_lsrej_post.sh $a $b > /root/mb/lspost.out 2>&1";;
  esac
  echo "[재개] $job $(date -u)"
done
tmux ls
