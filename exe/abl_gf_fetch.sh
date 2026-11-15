#!/bin/bash
# 절 CM183: GF 장면 (toast · sand_castle) 준비 -- exe/gfs_ours.sh 와 같은 /root/gfs 레이아웃 (GW). vast_abl.sh 가 source (cwd = 코드 디렉토리 $D; $R $MI $SC has() say 를 줌).
#   만드는 것: $GW/GaussianFluent (렌더 훅 --rr gf 용) · $GW/hf/{model/<이름>,config/<설정>} · $GW/ag/<이름>.pt · $GW/ag/gt_<이름> (정답 폴더, pg_h5_to_pt)
GW=/root/gfs; DSG=r2:storage/datasets/anchorflow/gf_scenes; mkdir -p $GW/hf/model $GW/hf/config $GW/ag $GW/gt
rclone copy $MI/GaussianFluent $GW/GaussianFluent --exclude "model/**" --exclude ".git/**" --exclude "demo/**" --exclude "static/**" --transfers 32 -q &
rclone copy $DSG/config $GW/hf/config -q &
for e in toast:tosta_config.json sand_castle:sandcastle_config.json; do n=${e%%:*}
  has $n || continue
  rclone copy $DSG/model/$n $GW/hf/model/$n --transfers 16 -q &
  rclone copy $DSG/gt/$n/simulation_ply $GW/gt/$n/simulation_ply --transfers 16 -q &
done
wait
pip install -q "urllib3<2.6" "brotli<1.2" 2>&1 | tail -1
for e in toast:tosta_config.json sand_castle:sandcastle_config.json; do n=${e%%:*}; c=${e##*:}
  has $n || continue
  [ -s $GW/ag/$n.pt ] && [ -s $GW/ag/gt_$n/config.json ] && continue
  O=$GW/ag/_$n; mkdir -p $O; CFG=$GW/hf/config/$c
  python - $CFG $O/cfg_floor.json $([ $n = sand_castle ] && echo CUT IMPULSE) <<'PYX'
import json, math, sys
d = json.load(open(sys.argv[1])); dx = d.get("grid_lim", 2.0) / d["n_grid"]; z3 = 3 * dx; hit = False
for b in d["boundary_conditions"]:
    if b["type"] == "surface_collider" and b.get("surface") == "sticky" and abs(b["normal"][2]) > 0.5:
        b["point"][2] = max(b["point"][2], z3); hit = True
if not hit and not any(t in sys.argv[3:] for t in ("NOFLOOR", "CUT")):
    d["boundary_conditions"].append({"type": "surface_collider", "point": [1, 1, z3], "normal": [0.0, 0.0, 1.0], "surface": "sticky", "friction": 0.0, "start_time": 0, "end_time": 1000.0})
_c = math.sqrt(d["E"] * (1 - d["nu"]) / ((1 + d["nu"]) * (1 - 2 * d["nu"]) * d["density"])); d["substep_dt"] = 0.6 * dx / _c
json.dump(d, open(sys.argv[2], "w"), indent=1)
PYX
  CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $GW/gt/$n/simulation_ply --config $O/cfg_floor.json --out_pt $GW/ag/$n.pt --out_dir $GW/ag/gt_$n --out_dir_n 1 --allow_nonfinite > $O/cv.log 2>&1 \
    || say "[GF] $n 변환 실패 $(tail -1 $O/cv.log | cut -c1-200)"
  say "[GF] $n pt $(stat -c %s $GW/ag/$n.pt 2>&1) gt $(ls $GW/ag/gt_$n 2>&1 | head -3 | tr '\n' ' ') 모델 $(ls $GW/hf/model/$n | head -3 | tr '\n' ' ')"
done
