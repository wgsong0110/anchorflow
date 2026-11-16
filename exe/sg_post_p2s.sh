#!/bin/bash
# Spring-Gaus pillow2sofa 학습 뒤 단계 (조정자 11-17: 단독 시간은 비어 있는 카드 ipg_breadb 에서): 단독 시간 1 회 -> 덤프 -> 물리 지표 ∥ (h5·렌더·시각 지표).
#   bash sg_post_p2s.sh P2  : (나) 근사 재개 판 (R2 bl/P2, 학습 ipg_wolft)   /  bash sg_post_p2s.sh P2X : (가) 정확 재학습 판 (R2 bl/P2X)
#   결과: R2 <판>/g/springgaus_pillow2sofa_n<N>_r1/{summary.json,res.json,card.txt,log.txt}, <판>/grid.json (pr2_grid collect 꼴), 지표 R2 final_fix/phys/<작업>/{phys,vis}.json + DONE
#   (작업 = P2: springgaus_pillow2sofa_nsg, P2X: springgaus_pillow2sofa_nsgx)
set -u
V=${1:?P2|P2X}
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
RA=r2:storage/result/anchorflow; RB=$RA/lbours/final_pa/bl; OX=$RB/$V; FX=$RA/lbours/final_fix; RP=$FX/phys
J=springgaus_pillow2sofa_nsg$([ $V = P2X ] && echo x)
O=/dev/shm/dkta/lbours/paper_remeasure2/stab; F=$O/p/springgaus_pillow2sofa_prep; C=/root/sgpost_code; NW=/home/dkta/work/newsc
L=/root/sgpost_$V.log; say() { echo "[sgpost $V $(date -u '+%F %T')] $*" >> $L; rclone copyto $L $OX/marks_mv/sgpost.log -q; }
mkdir -p /workspace/shm /home/dkta/.conda/envs/af/bin /home/dkta/bin $NW
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -x /home/dkta/.conda/envs/af/bin/python ] || ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
[ -x /home/dkta/bin/rclone ] || ln -sf $(command -v rclone) /home/dkta/bin/rclone
say "준비 $(hostname)"
[ -d /dev/shm/dkta/pylib/sg ] && [ -d /dev/shm/dkta/ext/Spring-Gaus ] || (cd /dev/shm/dkta && rclone cat $RB/code/sg_env_cf55da5.tar | tar -x)
[ -s $F/pillow2sofa.yaml ] || (mkdir -p $O/p && cd $O/p && rclone cat $RB/code/sg_fit_p2s.tar | tar -x)
[ -s $NW/pillow2sofa.pt ] || rclone copyto $RA/lbours/paper_remeasure2/scenes/pillow2sofa/pillow2sofa.pt $NW/pillow2sofa.pt -q
[ -d $NW/gt_pillow2sofa ] || rclone copy $RA/lbours/paper_remeasure2/scenes/pillow2sofa/gt_pillow2sofa $NW/gt_pillow2sofa -q --transfers 8
rm -rf $C; mkdir -p $C; rclone cat $RB/code/fpabl_cf55da5.tar | tar -x -C $C; rclone cat $FX/code/ffxphys_1717e403.tar | tar -x -C $C
cd $C; python exe/ffx_phys_patch_render.py >> $L 2>&1
python -c "import pyiqa" 2>/dev/null || pip install -q pyiqa -c <(pip freeze | grep -iE "^(torch|torchvision|numpy|scipy|scikit-image|opencv-python-headless)==") >> $L 2>&1
PPX=""; python -c "import open3d" 2>/dev/null || { mkdir -p /root/o3dstub/open3d; printf 'def __getattr__(n):\n    raise RuntimeError("open3d stub: " + n)\n' > /root/o3dstub/open3d/__init__.py; PPX=/root/o3dstub:; }
[ -d /root/ref/pillow2sofa ] || { mkdir -p /root/ref/pillow2sofa.tmp && rclone copy $RP/mpm_pillow2sofa/png /root/ref/pillow2sofa.tmp -q --transfers 16 && mv /root/ref/pillow2sofa.tmp /root/ref/pillow2sofa; }
say "준비 끝 (정답 렌더 $(ls /root/ref/pillow2sofa | wc -l)) -> 대기: R2 $V/MV_TRAIN_DONE"
until rclone lsf $OX/MV_TRAIN_DONE 2>/dev/null | grep -q . || rclone lsf $OX/DONE 2>/dev/null | grep -q .; do sleep 30; done
rclone copy $OX/p/springgaus_pillow2sofa_prep $F -q --max-depth 1 --include "sg_train.json"
EL=$(python -c "import json;print(json.load(open('$F/sg_train.json'))['exp'])"); rclone copy $OX/p/springgaus_pillow2sofa_prep/exp/$(basename $EL) $EL -q --exclude "evaluations/**" --exclude "dynamic_refine/**"
N=$(python -c "import json;print(json.load(open('$F/sg_train.json'))['n_step'])")
DA="--data $NW/pillow2sofa.pt --tag pillow2sofa --frames 300 --floor none --cfg_bc --pg_vdamp"
others() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v "^$"; }
say "학습 판 받음 ($EL, N $N) -> 카드 빌 때까지 대기 ($(others | wc -l) 프로세스)"
while [ -n "$(others)" ]; do sleep 10; done
G=$O/g$V; T=$G/springgaus_pillow2sofa_n${N}_r1; mkdir -p $T
echo "시작 때 카드 0 의 다른 프로세스 $(others | wc -l) 개 $(date -u '+%F %T') $(hostname) $(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader)" > $T/card.txt
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PPX}/dev/shm/dkta/pylib/sg python -u exe/stab_sg.py --mode run --sg /dev/shm/dkta/ext/Spring-Gaus --fit $F --N $N --out $T $DA > $T/log.txt 2>&1 &
P=$!; seen=""
while kill -0 $P 2>/dev/null; do seen="$seen $(others | grep -v "^$P$")"; sleep 5; done; wait $P; rc=$?
echo "rc=$rc 끝 $(date -u '+%F %T') 실행 중 다른 계산 프로세스: $(echo $seen | tr ' ' '\n' | sort -u | grep -v '^$' | tr '\n' ' ')" >> $T/card.txt
python - $T $G $N "$(echo $seen | tr ' ' '\n' | sort -u | grep -v '^$' | wc -l)" <<'P'
import json, sys
T, G, N, nf = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
s = json.load(open(f"{T}/summary.json")); dv = s.get("diverged")
r = dict(stable=not dv, err=(100 * s["err_mean"] if not dv else None), div=dv, ms=s.get("ms_mean"), ms_max=s.get("ms_max"), N=N, rep=1, others=max(nf - 1, 0))
json.dump(r, open(f"{T}/res.json", "w"), indent=1)
ok = [r] if r["stable"] and r["ms"] is not None else []
json.dump([dict(method="springgaus", scene="pillow2sofa", N=N, reps=1, ok=len(ok), err=r["err"] if ok else None, err_all=[q["err"] for q in ok], ms=r["ms"] if ok else None,
                ms_all=[q["ms"] for q in ok], div=[] if ok else [dv])], open(f"{G}/grid.json", "w"), indent=1)
print("시간 1 회", r)
P
rclone copy $G $OX/g -q; rclone copyto $G/grid.json $OX/grid.json -q
say "단독 시간 1 회 끝: $(cat $T/res.json | tr -d '\n ')"
# ---- 덤프 + 지표 ----
X=/workspace/fxp/$J; rm -rf $X; mkdir -p $X
echo "$(hostname) sgpost $(date)" | rclone rcat $RP/$J/LOCK -q
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PPX}/dev/shm/dkta/pylib/sg python -u exe/stab_sg.py --mode run --sg /dev/shm/dkta/ext/Spring-Gaus --fit $F --N $N --out $X $DA --dump --dump_F > $X/run.log 2>&1
say "덤프 rc=$? $(du -sh $X/pred.pt 2>&1 | cut -f1)"
( python -u exe/ffx_phys.py --method springgaus --scene pillow2sofa --N $N --data $NW/pillow2sofa.pt --pred $X/pred.pt --no_F --floor none --frames 300 --out $X/phys.json --note "Spring-Gaus 대표 설정 덤프 판 (N $N, $V)" > $X/phys.log 2>&1 ) &
PP=$!
python -u exe/lb_roll_to_h5.py --dump $X/pred.pt --traj $NW/gt_pillow2sofa --out $X/h5 > $X/roll_h5.log 2>&1
python -u exe/lb_render.py --sim $X/h5 --config $NW/gt_pillow2sofa/config.json --shape pillow2sofa --out $X/$J.mp4 --png $X/png --frames 300 < /dev/null > $X/render.log 2>&1
say "렌더 rc=$? $(ls $X/png 2>/dev/null | wc -l) 장"
python -u exe/ffx_vis.py --png $X/png --ref /root/ref/pillow2sofa --out $X/vis.json > $X/vis.log 2>&1
say "시각 rc=$? $(head -c 250 $X/vis.json 2>/dev/null)"
wait $PP; say "물리 $(head -c 250 $X/phys.json 2>/dev/null)"
rm -rf $X/h5
rclone copy $X $RP/$J -q --transfers 8 --exclude "/pred.pt" && [ -s $X/phys.json ] && [ -s $X/vis.json ] && touch $X/DONE && rclone copyto $X/DONE $RP/$J/DONE -q
rclone copyto $X/pred.pt $OX/g/springgaus_pillow2sofa_dump/pred.pt -q
say "끝 DONE $(rclone lsf $RP/$J/DONE 2>/dev/null)"
