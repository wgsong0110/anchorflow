#!/bin/bash
# final_fix_sg (마감 17일 11:00 KST): Spring-Gaus pillow2sofa 덤프 판 지표를 최대한 빨리. 준비 (코드·렌더 패치·정답 렌더·pyiqa 가중치) 를 지금 해 두고,
# 실제 pr2_grid (학습 + 단독 시간 r1, sgcut 가 r1 뒤 끝냄) 가 끝나면 곧바로: sg_train.json R2 로 -> 덤프 (같은 카드, 단독 시간 뒤라 겹침 없음) ->
# 물리 지표 (CPU, 뒤에서) 와 h5·렌더·시각 지표를 병렬 -> R2 phys/springgaus_pillow2sofa_nsg 올리고 DONE.
source /root/fbenv.sh; export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
L=/root/sg_p2s_fast.log; MK=r2:storage/result/anchorflow/lbours/final_fix/marks/final_fix_sg_p2sfast.log
say() { echo "[p2sfast $(date '+%F %T')] $*" >> $L; rclone copyto $L $MK -q; }
R=r2:storage/result/anchorflow/lbours; FX=$R/final_fix; RP=$FX/phys; J=springgaus_pillow2sofa_nsg
W=/home/dkta/work; NW=$W/newsc; PO=/dev/shm/dkta/lbours/paper_remeasure2/stab/p; F=$PO/springgaus_pillow2sofa_prep
D=/root/pr/code/cf55da5; cd $D
# ---- 준비 (지금) ----
rclone cat $FX/code/ffxphys_1717e403.tar | tar -x -C $D
python exe/ffx_phys_patch_render.py >> $L 2>&1
source /root/pr/bl_env.sh
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pr XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext CUDA_VISIBLE_DEVICES=0
[ -d /root/ref/pillow2sofa ] || { mkdir -p /root/ref/pillow2sofa.tmp && rclone copy $RP/mpm_pillow2sofa/png /root/ref/pillow2sofa.tmp -q --transfers 16 && mv /root/ref/pillow2sofa.tmp /root/ref/pillow2sofa; }
say "정답 렌더 $(ls /root/ref/pillow2sofa | wc -l) 장"
mkdir -p /workspace/vwarm/png; cp $(ls /root/ref/pillow2sofa/*.png | head -3) /workspace/vwarm/png/
python -u exe/ffx_vis.py --png /workspace/vwarm/png --ref /root/ref/pillow2sofa --out /workspace/vwarm/vis.json > /workspace/vwarm/vis.log 2>&1
say "pyiqa 예열 rc=$? $(tail -c 200 /workspace/vwarm/vis.log | tr '\n' ' ')"
say "준비 끝 -> 대기: 실제 pr2_grid 끝"
# ---- 학습·r1 끝 대기 ----
while pgrep -f "python -u exe/pr2_grid.py --out" >/dev/null; do sleep 20; done
[ -s $F/sg_train.json ] || { say "sg_train.json 없음 -- 학습 실패? 멈춤"; exit 1; }
rclone copyto $F/sg_train.json $R/final_pa/bl/P2/p/springgaus_pillow2sofa_prep/sg_train.json -q
rclone lsf $RP/$J/DONE 2>/dev/null | grep -q . && { say "이미 DONE"; exit 0; }
echo "$(hostname) p2sfast $(date)" | rclone rcat $RP/$J/LOCK -q
N=$(python -c "import json;print(json.load(open('$F/sg_train.json'))['n_step'])")
O=/workspace/fxp/$J; rm -rf $O; mkdir -p $O
DA="--data $NW/pillow2sofa.pt --tag pillow2sofa --frames 300 --floor none --cfg_bc --pg_vdamp"
say "덤프 시작 N $N (카드: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) 프로세스)"
PYTHONPATH=${ST_SG_PYPATH:-} python -u exe/stab_sg.py --mode run --sg ${ST_SG_REPO} --fit $F --N $N --out $O $DA --dump --dump_F > $O/run.log 2>&1
say "덤프 끝 rc=$? $(ls -la $O/pred.pt 2>&1 | awk '{print $5}') $(tail -c 200 $O/run.log | tr '\n' ' ')"
( python -u exe/ffx_phys.py --method springgaus --scene pillow2sofa --N $N --data $NW/pillow2sofa.pt --pred $O/pred.pt --no_F --floor none --frames 300 --out $O/phys.json --note "Spring-Gaus 대표 설정 덤프 판 (N $N)" > $O/phys.log 2>&1; echo "phys rc=$?" >> $L ) &
PP=$!
python -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $NW/gt_pillow2sofa --out $O/h5 > $O/roll_h5.log 2>&1
say "h5 rc=$?"
python -u exe/lb_render.py --sim $O/h5 --config $NW/gt_pillow2sofa/config.json --shape pillow2sofa --out $O/$J.mp4 --png $O/png --frames 300 < /dev/null > $O/render.log 2>&1
say "렌더 rc=$? $(ls $O/png 2>/dev/null | wc -l) 장"
python -u exe/ffx_vis.py --png $O/png --ref /root/ref/pillow2sofa --out $O/vis.json > $O/vis.log 2>&1
say "시각 rc=$? $(cat $O/vis.json 2>/dev/null | head -c 300)"
wait $PP
say "물리 $(cat $O/phys.json 2>/dev/null | head -c 300)"
rclone copy $O $RP/$J -q --transfers 8 --exclude "h5/**" --exclude "/pred.pt" && [ -s $O/phys.json ] && [ -s $O/vis.json ] && touch $O/DONE && rclone copyto $O/DONE $RP/$J/DONE -q
say "올림 끝 DONE $(rclone lsf $RP/$J/DONE 2>/dev/null)"
rclone copyto $O/pred.pt $R/final_pa/bl/P2/g/springgaus_pillow2sofa_dump/pred.pt -q
