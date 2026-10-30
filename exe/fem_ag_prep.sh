#!/bin/bash
# 절 AG 준비 (잡 안, 카드 AG_G 하나 -- 다른 갈래가 나머지를 쓰는 동안): (0) 정답 h5 -> .pt (형식만, 노드 /tmp, R2 lbours/fem_ag_data)
# (1) 단위 대조 (정답 원본 warp 커널 vs fem_pl_cuda) (2) 탄성 판 불변 대조 (lego L000 5 프레임, --plast 유무 -- jelly 는 같은 식이어야) (3) 씬 연기 시험 3 프레임.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
source exe/fem_ag_scene.sh
G=${AG_G:-1}; OD=/tmp/dkta/lbours/fem_ag_prep; RR=r2:storage/result/anchorflow/lbours; mkdir -p $OD $AGD
echo "[커밋] $(git log -1 --oneline) $(date '+%F %T') 카드 $G"; df -h /tmp | tail -1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# (0) 변환
cv() { local n=$1; shift; [ -f $AGD/$n.pt ] || CUDA_VISIBLE_DEVICES=$G python -u exe/pg_h5_to_pt.py "$@" --out_pt $AGD/$n.pt > $AGD/cv_$n.log 2>&1; tail -3 $AGD/cv_$n.log; }
cv hotdog --h5_dir $W/topo/pg_hotdog_visco --config $W/wmats/hotdog_viscoplastic.json --out_dir $AGD/gt_hotdog &
cv wolf --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_dir $AGD/gt_wolf &
cv bread --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_dir $AGD/gt_bread &
cv wm --h5_dir $W/wm2/gf/simulation_ply --config $W/wm2/gf/config.json --out_dir $AGD/gt_wm --out_dir_n 1 --allow_nonfinite &
wait
CUDA_VISIBLE_DEVICES=$G python - <<'PY' > $OD/gt_stats.txt 2>&1
import torch
for n in ["hotdog", "wolf", "bread", "wm"]:
    d = torch.load(f"/tmp/dkta/ag_data/{n}.pt", weights_only=False)
    x = d["x"].float(); ok = torch.isfinite(x).all(-1)
    xs = torch.where(ok[..., None], x, torch.nan)
    lo = torch.nan_to_num(xs, nan=9e9).min(1).values; hi = torch.nan_to_num(xs, nan=-9e9).max(1).values
    v0 = d["v"][0].float()
    print(n, "프레임", x.shape[0], "입자", x.shape[1], "비유한(마지막)", int((~ok[-1]).sum()), "0프레임 v 평균", [round(q, 4) for q in v0.mean(0).tolist()],
          "|v0| 최대", round(float(v0.norm(dim=-1).max()), 4), "F0-I", float((d["F"][0].float() - torch.eye(3)).abs().max()), "F1 범위", float(d["F"][1].float().abs().max()))
    for i in [0, 1, 2, 5, 10, x.shape[0] // 2, x.shape[0] - 1]:
        print("   프레임", i, "최소", [round(q, 4) for q in lo[i].tolist()], "최대", [round(q, 4) for q in hi[i].tolist()])
    print("   cfg", {k: d["cfg"].get(k) for k in ("material", "E", "nu", "density", "yield_stress", "plastic_viscosity", "friction_angle", "hardening", "xi", "beta", "n_grid", "grid_lim", "frame_dt", "substep_dt", "g", "grid_v_damping_scale", "rpic_damping")})
PY
cat $OD/gt_stats.txt
rclone copy $AGD $RR/fem_ag_data -q --exclude "gt_*/**" &
# (1) 단위 대조
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_ag_unit.py --solver pg --out $OD/unit_pg.json > $OD/unit_pg.log 2>&1; cat $OD/unit_pg.log | grep -v Warp | tail -30
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_ag_unit.py --solver gf --out $OD/unit_gf.json > $OD/unit_gf.log 2>&1; cat $OD/unit_gf.log | grep -v Warp | tail -10
# (2) 탄성 판 불변 (jelly: --plast 이 있어도 같은 식) -- lego L000 5 프레임
for P in base plast; do
  X=""; [ $P = plast ] && X="--plast"
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_ex_sim.py --out $OD/par_$P --frames 5 --floor vel --N 256 --no_dump $REC $X < /dev/null > $OD/par_$P.log 2>&1
  grep -a "롤아웃\]" $OD/par_$P.log
done
python - <<PY
import json
a = [json.loads(l)["err"] for l in open("$OD/par_base/fem.jsonl")]; b = [json.loads(l)["err"] for l in open("$OD/par_plast/fem.jsonl")]
print("[탄성 불변] 프레임별 오차 기준", a, "\n            --plast   ", b, "\n            최대 차", max(abs(x - y) for x, y in zip(a, b)))
PY
# (3) 씬 연기 시험 (3 프레임)
for S in hotdog wolf bread wm; do
  ag_scene $S
  N=256; [ $S = wolf ] && N=1024; [ $S = wm ] && N=1024; [ $S = bread ] && N=64
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_ex_sim.py --out $OD/smoke_$S $SARGS --frames 3 --no_dump $REC --N $N < /dev/null > $OD/smoke_$S.log 2>&1
  echo "== $S N$N"; grep -a "\[CFL\]\|절 AG\|정답\]\|롤아웃\]\|발산\|Error\|error" $OD/smoke_$S.log | cut -c1-400; grep -a "프레임 [0-9]*/" $OD/smoke_$S.log | cut -c1-260
done
wait
rclone copy $OD $RR/fem_ag_prep -q
echo "AG_PREP_DONE $(date +%T)"; touch $OD/AG_PREP_DONE
