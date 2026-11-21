#!/bin/bash
# sg_resume 검증 2 차 (B·L 다시 -- cfg defrost 고침, E = empty_cache 끔; A1·A2 는 1 차 그대로). 원래: 작은 설정: A1·A2 = 한 번에 8 반복 (비결정성 수준), B = 반복 4 저장 뒤 끊고 이어 돌림 (새 형식), L = 옛 체크포인트 (공식 SAVE_INTERVAL 4) 에서 이어 돌림
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
export ST_SG_REPO=/dev/shm/dkta/ext/Spring-Gaus ST_SG_PYPATH=/dev/shm/dkta/pylib/sg
Z=/workspace/sgrtest; SRC=/dev/shm/dkta/lbours/paper_remeasure2/stab/p/springgaus_pillow2sofa_prep
mk() { local F=$Z/$1; rm -rf $F; mkdir -p $F; cp $SRC/af_bc.json $SRC/af_v0.pt $SRC/af_xf.json $F/
  python - $SRC/pillow2sofa.yaml $F/pillow2sofa.yaml <<'P'
import sys; s=open(sys.argv[1]).read()
s=s.replace("DYNAMIC:\n", "DYNAMIC:\n  ITERATIONS: 8\n  ITER_REFINE: 20\n  N_STEP: 100\n  STEP_ADD: 50\n  N_STEP_ITER: [3, 6]\n  MAX_N_STEP: 500\n  SAVE_INTERVAL: 4\n",1)
import re; s=re.sub(r"N_FRAME: \d+", "N_FRAME: 31", s); s=re.sub(r"EVAL_FRAME: \d+", "EVAL_FRAME: 31", s); s=re.sub(r"HIT_FRAME: \d+", "HIT_FRAME: 30", s)
open(sys.argv[2],"w").write(s)
P
}
run() { local F=$Z/$1; shift; cd $C; env "$@" CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1 OMP_NUM_THREADS=8 SG_RESUME_EVERY=2 PYTHONPATH=$ST_SG_PYPATH python -u exe/stab_sg.py --mode train --sg $ST_SG_REPO --fit $F --out $F \
   --data /home/dkta/work/newsc/pillow2sofa.pt --tag pillow2sofa --frames 300 --floor none --cfg_bc --pg_vdamp >> $F/train.log 2>&1; echo "rc=$?" >> $F/train.log; }
C=$Z/code; cp $Z/ov/exe/* $C/exe/
for x in B E; do mk $x; done
( run B SG_RESUME_STOP_AT=4; mkdir -p $Z/L; cp -r $Z/B/. $Z/L/; rm -rf $Z/L/sg_resume; : > $Z/L/train.log; ( run L X=1 ) & run B X=1; wait ) &
run E SG_NO_EMPTY_CACHE=1
wait
python - $Z <<'P'
import glob, sys, torch, numpy as np, re
from plyfile import PlyData
Z = sys.argv[1]
def fin(x):
    e = sorted(glob.glob(f"{Z}/{x}/exp/*/checkpoints_dynamic/checkpoint/Spring_Mass.pth.tar"))[-1]
    sd = torch.load(e, map_location="cpu"); d = e.rsplit("/", 1)[0]
    p = PlyData.read(d + "/gaussians.ply").elements[0]
    g = np.stack([np.asarray(p[n]) for n in p.data.dtype.names], 1)
    ls = re.findall(r"frame: (\d+)/\d+ cam: \d+/\d+ loss: ([0-9.]+)", open(f"{Z}/{x}/train.log").read())
    return sd, g, open(d + "/dy_n_step.json").read().strip()
R = {x: fin(x) for x in ("A1", "A2", "B", "L", "E")}
for x in ("A2", "B", "L", "E"):
    sd, g, n = R[x]; sa, ga, na = R["A1"]
    dp = {k: float((sd[k].double() - sa[k].double()).abs().max()) for k in sa}
    print(f"{x} vs A1: n_step {n}/{na}, 매개변수 최대 차 {max(dp.values()):.3g} ({ {k: '%.3g' % v for k, v in dp.items() if v > 0} }), 가우시안 ply 최대 차 {np.abs(g - ga).max():.3g}")
print("A1 값", {k: [round(float(q), 6) for q in v.flatten()[:3]] for k, v in R["A1"][0].items()})
P
