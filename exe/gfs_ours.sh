#!/bin/bash
# 절 CM182: GF 공개 장면 하나에 우리 판 (기본 판 K 1, 수박 명령 = exe/stab_k1new.sh 의 M 그대로, 씬마다 데이터·프레임 수·바닥만) + 위상·시각 지표 + 영상.
#   사용: bash gfs_ours.sh 이름 모델 설정 프레임수 바닥(plane|cfg)   (vast_gfs.sh 가 W AF DS R RX GF TAG 와 say 를 준 뒤 부름)
#   바닥: plane = GF 격자 경계 padding 3 칸 (z = 3 dx, 설정의 z=0 sticky 평면은 GF 안에서 작동 안 함) -> --floor_plane 1,1,<3dx>,0,0,1, cfg = 설정 sticky 평면 (z > 3dx 라 실제로 작동) 그대로.
n=$1; m=$2; c=$3; NF=$4; XEXTRA=${5:-}; XEXTRA=${XEXTRA//+/ }
ABL=""; for w in $XEXTRA; do case $w in ABL0|ABLA|ABLB|ABLC) ABL=$w;; esac; done      # 원인 분리 실험 (사용자 11-15): 0 기본, A 초기 속도 0, B 재질 탄성, C 바닥 미끄럼; 프레임 NF 까지만, 지표·영상 없음
SFX=${XS_SFX:-}; [ -n "$ABL" ] && SFX=${XS_SFX:-_$(echo $ABL | tr A-Z a-z)}; O=$W/ours/$n$SFX; mkdir -p $O; CFG=$W/hf/config/$c; MODEL=$W/hf/model/$m; D=$DS/gt/$n
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PYTHONPATH=$GF:$GF/gaussian-splatting
cd $AF
rclone copy $D/simulation_ply $W/gt/$n/simulation_ply --transfers 16 -q; NH=$(ls $W/gt/$n/simulation_ply | grep -c h5)
say "  [$n] 정답 h5 $NH 개 (config 프레임 $NF + 1)"
# 바닥: GF 는 격자 경계 padding 3 칸 (z = 3 dx) 이 실제 바닥이고 config 의 z=0 sticky 평면은 작동 안 한다 -> 우리 판이 읽는 .pt 안 config 사본의 sticky 평면 높이를 max(원래, 3 dx) 로 (정답 시뮬 config 는 그대로, 바닥만 바꿈)
python - $CFG $O/cfg_floor.json $XEXTRA <<'PYX'
import json, sys
d = json.load(open(sys.argv[1])); dx = d.get("grid_lim", 2.0) / d.get("n_grid", 50); z3 = 3 * dx; hit = False
for b in d["boundary_conditions"]:
    if b["type"] == "surface_collider" and b.get("surface") in ("sticky", "cut") and abs(b["normal"][2]) > 0.5:
        if b["surface"] == "cut" and b["point"][2] < z3:      # lollipop: cut 평면 z=0 -> 격자 노드 (z*dx >= 0) 중 평면 아래가 없어 GF 에서 아무 일도 안 함
            continue
        # cut (GF surface_type 11): 평면 아래 노드 속도를 매 서브스텝 0 (z<0.4) 또는 x0.3·y0 (슬랩) -- 프레임당 서브스텝 126 이면 슬랩도 0.3^126 = 0 이라 프레임 단위에서 sticky 와 같다 -> 사본 config 에서 sticky 로
        b["surface"] = "sticky"; b["point"][2] = max(b["point"][2], z3); hit = True
if not hit and "NOFLOOR" not in sys.argv[3:]:
    d["boundary_conditions"].append({"type": "surface_collider", "point": [1, 1, z3], "normal": [0.0, 0.0, 1.0], "surface": "sticky", "friction": 0.0, "start_time": 0, "end_time": 1000.0})
# GF 러너는 config 의 substep_dt 를 무시하고 0.6 dx / 음속 으로 다시 계산한다 (gaussianfluent-hidden-defaults 2번) -> 우리 판의 격자 속도 감쇠 (--pg_vdamp, 서브스텝당 grid_v_damping_scale) 지수가 GF 실제 서브스텝 수와 맞도록 사본 config 의 substep_dt 를 GF 실제값으로
import math
_c = math.sqrt(d["E"] * (1 - d["nu"]) / ((1 + d["nu"]) * (1 - 2 * d["nu"]) * d["density"])); d["substep_dt"] = 0.6 * dx / _c
print("substep_dt 원래 config", json.load(open(sys.argv[1]))["substep_dt"], "-> GF 실제", d["substep_dt"], "서브스텝/프레임", int(d["frame_dt"] / d["substep_dt"]))
if "ABLB" in sys.argv[3:]: d["material"] = "jelly"      # 실험 B: 같은 E·ν 의 탄성
if "ABLC" in sys.argv[3:]:                                # 실험 C: 바닥을 sticky 대신 미끄럼 (마찰 0)
    for b in d["boundary_conditions"]:
        if b["type"] == "surface_collider" and b.get("surface") == "sticky": b["surface"] = "slip"; b["friction"] = 0.0
json.dump(d, open(sys.argv[2], "w"), indent=1); print("바닥 z", [b["point"][2] for b in d["boundary_conditions"] if b["type"] == "surface_collider" and b.get("surface") == "sticky"])
PYX
H5D=$W/gt/$n/simulation_ply
if [ -n "$ABL" ]; then H5D=$W/gt/$n/sim_abl; mkdir -p $H5D; for i in $(seq 0 $NF); do ln -sf $W/gt/$n/simulation_ply/sim_$(printf %010d $i).h5 $H5D/sim_$(printf %010d $i).h5; done; fi
python -u exe/pg_h5_to_pt.py --h5_dir $H5D --config $O/cfg_floor.json --out_pt $W/ag/$n.pt --out_dir $W/ag/gt_$n --out_dir_n 1 --allow_nonfinite > $O/cv.log 2>&1 || { say "  [$n] 변환 실패 $(tail -1 $O/cv.log | cut -c1-200)"; exit 1; }
if [ "$ABL" = ABLA ]; then python - $W/ag/$n.pt <<'PYX'
import sys, torch
d = torch.load(sys.argv[1], weights_only=False); d["v"][0] = 0; torch.save(d, sys.argv[1]); print("실험 A: 0 프레임 속도 0")
PYX
fi
mkdir -p /home/dkta/.conda/envs/af/bin; ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
rclone cat $R/lbours/wolfcv/cm157/BA_1/cmd.txt > $O/BA_1.txt
PRE=""; FLR=1; XE=""
for w in $XEXTRA; do case $w in CUT) ;; HFF) ;; IMPULSE) PRE="$PRE exe/impulse_wrap.py --impulse_bc 1";; NOFLOOR) FLR=0;; ABL0|ABLA|ABLB|ABLC) ;; OLD) XE="$XE";; *) XE="$XE $w";; esac; done
CMD=$(GFS_PREFIX="$PRE" python exe/gfs_cmd.py $O/BA_1.txt $W/ag/$n.pt $NF $n $O/run $FLR $W/ag/gt_$n $CFG $MODEL $GF $XE)
if [[ " $XEXTRA " == *" OLD "* ]]; then      # 후보 2 시험 (사용자 11-15): 뒤집힘 막는 선탐색 (--ls_inv ...) 이 있는 옛 재질 실행기 (수박 M 판, stab_k1new.sh) 의 명령. 바닥 평면 = 3 dx
  FP="--floor_plane 1,1,$(python -c "import json;d=json.load(open('$CFG'));print(3*d.get('grid_lim',2.0)/d['n_grid'])"),0,0,1"
  CMD="python -u exe/fem_ao_rt_sim.py --data $W/ag/$n.pt --frames $NF --floor vel $FP --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag $n --K 1 --dump --out $O/run"
fi
echo "$CMD" > $O/cmd.txt; rclone copyto $O/cmd.txt $D/ours_cmd.txt -q
say "  [$n] 우리 판 시작: ${CMD:0:260}"
mkdir -p $O/run; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 timeout 14400 bash -c "$CMD" < /dev/null > $O/run.log 2>&1; rc=$?
say "  [$n] 우리 판 rc=$rc $(( $(date +%s) - t0 )) 초 $(grep -ah '롤아웃\]' $O/run.log | tail -1 | cut -c1-240) $(grep -a 'rror\|발산' $O/run.log | tail -1 | cut -c1-200)"
if [ -n "${XS_SWEEP:-}" ] && [ -z "${XS_SWEEP_LAST:-}" ] && python -c "import json,sys;s=json.load(open('$O/run/summary.json'));sys.exit(0 if s.get('diverged') else 1)" 2>/dev/null; then
  say "  [$n] K 시험 ($SFX): 발산 -> 다음 K"; rclone copy $O/run $RX/ours/$n$SFX -q --exclude "pred.pt" --transfers 8; exit 3; fi
rclone copy $O/run $RX/ours/$n$SFX -q --exclude "pred.pt" --transfers 8; rclone copyto $O/run.log $RX/ours/$n$SFX/run.log -q
[ -s $O/run/pred.pt ] || { say "  [$n] pred.pt 없음"; exit 1; }
if [ -n "$ABL" ]; then python - $O/run/fem.jsonl <<'PYX' | tee -a $LOG
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])]; o = []
for x in r:
    pd, qj = x["pd"], x["qj"]; o.append("f%d Jmin %.2e nJneg %d qneg %d err %.2f%% hpmax %.2f" % (x["t"] + 1, pd["Jmin"], pd["nJneg"], qj["nneg"], 100 * x["err"], x["hp"][1]))
print("[실험 요약] " + " | ".join(o[::2][:12]))
PYX
  say "  [$n] 실험 $ABL 끝"; exit 0; fi
python -u exe/gfs_post.py topo --gt_pt $W/ag/$n.pt --pred $O/run/pred.pt --out $O/topo.json > $O/topo.log 2>&1; say "  [$n] $(tail -1 $O/topo.log | cut -c1-500)"
# ---- 렌더 (정답 h5 | 우리) 같은 GF 렌더러·카메라, 배경 없음. 우리 = pred 를 정답 h5 형식으로 (발산 뒤는 마지막 유한 프레임에서 정지)
python - $O/run/pred.pt $O/run/pred_v.pt $NF <<'PYX' > $O/pad.log 2>&1
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False); P, F = d["pred"], d["F"]; L = int(sys.argv[3]) + 1
n = 0
while n < P.shape[0] and bool(torch.isfinite(P[n]).all()) and bool(torch.isfinite(F[n]).all()): n += 1
P, F = P[:n], F[:n]
if n < L: P = torch.cat([P, P[n - 1:n].expand(L - n, -1, -1)]); F = torch.cat([F, F[n - 1:n].expand(L - n, -1, -1, -1)])
torch.save({"pred": P.contiguous(), "F": F.contiguous()}, sys.argv[2]); print("유한 프레임", n, "->", P.shape[0])
PYX
python -u exe/lb_roll_to_h5.py --dump $O/run/pred_v.pt --traj $W/ag/gt_$n --out $O/h5_ours > $O/h5.log 2>&1
rw() { (cd $GF && python -u $AF/exe/gfs_render.py --gf $GF --model_path $MODEL --config $CFG --frames $((NF + 1)) "$@" < /dev/null); }
rw --sim $W/gt/$n/simulation_ply --out $O/gt.mp4 > $O/r_gt.log 2>&1; rw --sim $O/h5_ours --out $O/ours.mp4 > $O/r_ours.log 2>&1
say "  [$n] 렌더 정답 PNG $(ls $O/gt_frames 2>/dev/null | wc -l) 우리 PNG $(ls $O/ours_frames 2>/dev/null | wc -l) $(grep -a 'rror' $O/r_gt.log $O/r_ours.log | tail -1 | cut -c1-200)"
python -u exe/gfs_post.py vid --gt_png $O/gt_frames --our_png $O/ours_frames --out_json $O/visual.json --out_mp4 $O/gf_${n}${SFX}_cmp_raw.mp4 --font $W/font.otf --fps 24 > $O/vid.log 2>&1; say "  [$n] $(tail -1 $O/vid.log | cut -c1-300)"
ffmpeg -y -loglevel error -i $O/gf_${n}${SFX}_cmp_raw.mp4 -vf "scale='min(2400,iw)':-2,format=yuv420p" -c:v libx264 -crf 18 -movflags +faststart $O/gf_${n}${SFX}_cmp.mp4
ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv=p=0 $O/gf_${n}${SFX}_cmp.mp4 | tee -a $LOG
rclone copyto $O/gf_${n}${SFX}_cmp.mp4 $R/v20261009/gf_${n}${SFX}_cmp.mp4 -q
rclone copy $O $RX/ours/$n$SFX -q --include "*.json" --include "*.log" --include "cmd.txt"; say "  [$n] 영상 $R/v20261009/gf_${n}${SFX}_cmp.mp4"
