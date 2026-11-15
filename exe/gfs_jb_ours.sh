#!/bin/bash
# 절 CM182: 젤리+총알 -- 공식 러너와 같은 프레임 일정 (시뮬은 프레임 0~19, 200~239, 381 이후만, 나머지는 상태 정지) 으로 우리 판을 돌리고 정답과 520 프레임 전부 같은 인덱스로 비교.
#   정답 h5 중 시뮬이 도는 프레임 (파일 0, 그리고 f+1, f in 활성) 만 모아 199 프레임짜리 압축 궤적으로 우리 판을 돌리고, 정지 프레임은 직전 상태를 되풀이해 521 상태로 펼친다 (정답 h5 도 정지 프레임은 같은 상태).
#   렌더 = 공식 젤리 러너의 렌더 경로 (배경 없음) 에 정답 h5 / 우리 h5 를 먹임 (patch_gf_jelly.py --traj).  사용: bash gfs_jb_ours.sh [추가 인자]  (vast_gfs.sh 가 W AF DS R RX GF TAG say 를 줌)
n=jellybullet; NF=199; XEXTRA=${1:-}; XEXTRA=${XEXTRA//+/ }
SFX=${XS_SFX:-}; O=$W/ours/$n$SFX; mkdir -p $O; CFG=$W/hf/config/jelly_config_nacc.json; MODEL=$W/hf/model/jelly; D=$DS/gt/$n
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PYTHONPATH=$GF:$GF/gaussian-splatting
cd $AF
rclone copy $D/simulation_ply $W/gt/$n/simulation_ply --transfers 16 -q; NH=$(ls $W/gt/$n/simulation_ply | grep -c h5); say "  [$n] 정답 h5 $NH 개 (521 기대)"
python - $W/gt/$n/simulation_ply $W/gt/$n/sim_c $O/map.json <<'PYX'
import json, os, sys
src, dst, mp = sys.argv[1:4]; os.makedirs(dst, exist_ok=True)
act = list(range(0, 20)) + list(range(200, 240)) + list(range(381, 520))
files = [0] + [f + 1 for f in act]
for i, f in enumerate(files):
    p = os.path.join(dst, "sim_%010d.h5" % i)
    if not os.path.lexists(p): os.symlink(os.path.join(src, "sim_%010d.h5" % f), p)
# 정답 상태 인덱스 j (0..520) -> 압축 인덱스 c(j) = 0..j-1 프레임 중 활성 프레임 수
cnt = [0]
for j in range(1, 521): cnt.append(cnt[-1] + (1 if (j - 1) in set(act) else 0))
json.dump(dict(act=act, files=files, c=cnt), open(mp, "w")); print("압축", len(files), "상태, 펼침", len(cnt), "상태 (c 끝", cnt[-1], ")")
PYX
python - $CFG $O/cfg_floor.json $XEXTRA <<'PYX'
import json, math, sys
d = json.load(open(sys.argv[1])); dx = d.get("grid_lim", 2.0) / d["n_grid"]; z3 = 3 * dx; hit = False
for b in d["boundary_conditions"]:
    if b["type"] == "surface_collider" and b.get("surface") == "sticky" and abs(b["normal"][2]) > 0.5:
        b["point"][2] = max(b["point"][2], z3); hit = True
_c = math.sqrt(d["E"] * (1 - d["nu"]) / ((1 + d["nu"]) * (1 - 2 * d["nu"]) * d["density"])); d["substep_dt"] = 0.6 * dx / _c
print("substep_dt 원래", json.load(open(sys.argv[1]))["substep_dt"], "-> GF 실제", d["substep_dt"], "서브스텝/프레임", int(d["frame_dt"] / d["substep_dt"]))
json.dump(d, open(sys.argv[2], "w"), indent=1)
PYX
python -u exe/pg_h5_to_pt.py --h5_dir $W/gt/$n/sim_c --config $O/cfg_floor.json --out_pt $W/ag/$n.pt --out_dir $W/ag/gt_$n --out_dir_n 1 --allow_nonfinite > $O/cv.log 2>&1 || { say "  [$n] 변환 실패 $(tail -1 $O/cv.log | cut -c1-200)"; exit 1; }
mkdir -p /home/dkta/.conda/envs/af/bin; ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
rclone cat $R/lbours/wolfcv/cm157/BA_1/cmd.txt > $O/BA_1.txt
CMD=$(python exe/gfs_cmd.py $O/BA_1.txt $W/ag/$n.pt $NF $n $O/run 1 $W/ag/gt_$n $CFG $MODEL $GF $XEXTRA)
echo "$CMD" > $O/cmd.txt; rclone copyto $O/cmd.txt $D/ours_cmd.txt -q
say "  [$n] 우리 판 시작 (압축 $NF 프레임): ${CMD:0:200}"
mkdir -p $O/run; t0=$(date +%s)
CUDA_VISIBLE_DEVICES=0 timeout 18000 bash -c "$CMD" < /dev/null > $O/run.log 2>&1; rc=$?
say "  [$n] 우리 판 rc=$rc $(( $(date +%s) - t0 )) 초 $(grep -ah '롤아웃\]' $O/run.log | tail -1 | cut -c1-240) $(grep -a 'rror\\|발산' $O/run.log | tail -1 | cut -c1-200)"
rclone copy $O/run $RX/ours/$n$SFX -q --exclude "pred.pt" --transfers 8; rclone copyto $O/run.log $RX/ours/$n$SFX/run.log -q
[ -s $O/run/pred.pt ] || { say "  [$n] pred.pt 없음"; exit 1; }
python -u exe/gfs_post.py topo --gt_pt $W/ag/$n.pt --pred $O/run/pred.pt --out $O/topo.json > $O/topo.log 2>&1; say "  [$n] $(tail -1 $O/topo.log | cut -c1-500)"
# 펼친 오차 (520 프레임, 정답과 같은 인덱스: 정지 프레임은 직전 활성 프레임의 오차를 되풀이)
python - $O/run/fem.jsonl $O/map.json $O/err_full.json <<'PYX'
import json, sys
e = [json.loads(l)["err"] for l in open(sys.argv[1])]; m = json.load(open(sys.argv[2]))["c"]
full = [e[m[j] - 1] if m[j] > 0 else 0.0 for j in range(1, 521)]
json.dump(dict(err_full=full, mean_full=sum(full) / len(full), mean_active=sum(e) / len(e), n_active=len(e)), open(sys.argv[3], "w")); print("[오차] 520 프레임 평균 %.3f%% 활성 %d 프레임 평균 %.3f%%" % (100 * sum(full) / len(full), len(e), 100 * sum(e) / len(e)))
PYX
say "  [$n] $(tail -1 $O/err_full.json 2>/dev/null | cut -c1-10)$(python -c "import json;d=json.load(open('$O/err_full.json'));print('오차 520 프레임 평균 %.3f%% (활성 %d 프레임 %.3f%%)' % (100*d['mean_full'], d['n_active'], 100*d['mean_active']))")"
# 우리 h5 (압축) -> 펼침 521
python - $O/run/pred.pt $O/run/pred_v.pt <<'PYX'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False); P, F = d["pred"], d["F"]
n = 0
while n < P.shape[0] and bool(torch.isfinite(P[n]).all()) and bool(torch.isfinite(F[n]).all()): n += 1
L = P.shape[0]
if n < L: P = torch.cat([P[:n], P[n - 1:n].expand(L - n, -1, -1)]); F = torch.cat([F[:n], F[n - 1:n].expand(L - n, -1, -1, -1)])
torch.save({"pred": P.contiguous(), "F": F.contiguous()}, sys.argv[2]); print("유한 프레임", n, "/", L)
PYX
python -u exe/lb_roll_to_h5.py --dump $O/run/pred_v.pt --traj $W/ag/gt_$n --out $O/h5_c > $O/h5.log 2>&1
python - $O/h5_c <<'PYX' >> $O/h5.log 2>&1
# 절 CM184: GF 젤리 렌더러 (gs_jb_traj) 는 GF h5 배치 (x (3,N), f_tensor (9,N)) 를 읽음 -- 우리 h5 (x (N,3), F (N,9)) 에 같은 값의 GF 배치 키를 더함 (좌표는 정답 h5 와 같음 확인)
import glob, sys, h5py, numpy as np
n = 0
for f in sorted(glob.glob(sys.argv[1] + "/sim_*.h5")):
    with h5py.File(f, "r+") as h:
        if "f_tensor" in h: continue
        x = np.array(h["x"]); F = np.array(h["F"]); del h["x"]
        h.create_dataset("x", data=np.ascontiguousarray(x.T)); h.create_dataset("f_tensor", data=np.ascontiguousarray(F.T))
        h.create_dataset("v", data=np.zeros_like(x.T)); h.create_dataset("C", data=np.zeros_like(F.T)); h.create_dataset("time", data=np.zeros((1, 1))); n += 1
print("GF 배치 키 더함", n)
PYX
python - $O/h5_c $O/h5_full $O/map.json <<'PYX'
import json, os, sys
src, dst, mp = sys.argv[1:4]; os.makedirs(dst, exist_ok=True); c = json.load(open(mp))["c"]
for j in range(0, 521):
    p = os.path.join(dst, "sim_%010d.h5" % j)
    if not os.path.lexists(p): os.symlink(os.path.join(src, "sim_%010d.h5" % c[j]), p)
print("펼침", len(os.listdir(dst)))
PYX
python $AF/exe/patch_gf_jelly.py $GF/gs_simulation/jelly/gs_simulation_jellynacc.py $GF/gs_jb_traj.py --mode nobg --dropbig --traj
cd $GF; cp $W/gt/$n/simulation_ply/../../../hf/config/jelly_config_nacc.json /dev/null 2>&1
rclone copyto $D/save_bullet_prop.pt save_bullet_prop.pt -q
for k in gt ours; do
  if [ $k = gt ]; then TD=$W/gt/$n/simulation_ply; else TD=$O/h5_full; fi
  GFS_TRAJ=$TD timeout 7000 python -u gs_jb_traj.py --model_path $MODEL --config $CFG --output_path $O/rend_$k --render_img < /dev/null > $O/rend_$k.log 2>&1
  say "  [$n] 렌더 $k PNG $(ls $O/rend_$k/*.png 2>/dev/null | wc -l) $(grep -a 'rror' $O/rend_$k.log | tail -1 | cut -c1-200)"
done
cd $AF
python -u exe/gfs_post.py vid --gt_png $O/rend_gt --our_png $O/rend_ours --out_json $O/visual.json --out_mp4 $O/gf_${n}_cmp_raw.mp4 --font $W/font.otf --fps 28 > $O/vid.log 2>&1; say "  [$n] $(tail -1 $O/vid.log | cut -c1-300)"
ffmpeg -y -loglevel error -i $O/gf_${n}_cmp_raw.mp4 -vf "scale='min(2400,iw)':-2,format=yuv420p" -c:v libx264 -crf 18 -movflags +faststart $O/gf_${n}_cmp.mp4
ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv=p=0 $O/gf_${n}_cmp.mp4 | tee -a $LOG
rclone copyto $O/gf_${n}_cmp.mp4 $R/v20261009/gf_${n}${SFX}_cmp.mp4 -q
rclone copy $O $RX/ours/$n$SFX -q --include "*.json" --include "*.log" --include "cmd.txt"; say "  [$n] 영상 $R/v20261009/gf_${n}${SFX}_cmp.mp4"
