#!/bin/bash
# allmodels 분석 도구 연기 시험 (가짜 작은 궤적): gtcheck · failmode · report(err/fail) · grid. 학습·롤아웃 없음.
#   잡 안에서만: AF_DETACH=1 bash exe/job_run.sh <이 스크립트> <로그>  (인자 없이; GPU 는 빈 카드 하나, 0 번 제외)
export PYTHONUTF8=1 OMP_NUM_THREADS=4 LC_ALL=C.UTF-8
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
cd "$(dirname "$0")/.." || exit 1
ME=$(pwd); C=$(git rev-parse --short HEAD); S=$W/lbours/allmodels/smoke_$C; mkdir -p $S; cd $S || exit 1
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
for c in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$1!=0 && $2<1024{print $1}'); do
  u=$(nvidia-smi -i $c --query-gpu=uuid --format=csv,noheader); grep -q "$u" <<<"$apps" || { G=$c; break; }; done
[ -z "$G" ] && { echo "빈 GPU 없음"; echo SMOKE_FAIL; exit 1; }
export CUDA_VISIBLE_DEVICES=$G; echo "[연기 시험] 커밋 $C GPU$G $(date)"
python - <<'PY'
import json, os, h5py, numpy as np, torch, imageio.v2 as imageio
W = "/home/dkta/work"
P = torch.load(f"{W}/lbpt/test/0004.pt", map_location="cpu", weights_only=False)
cfg = P["cfg"]; del P
n, L = 3000, 20
torch.save({"cfg": cfg, "sel": np.arange(n)}, "fake.pt")
rng = np.random.default_rng(0)
bc = [b for b in cfg.get("boundary_conditions", []) if b.get("type") == "surface_collider"]
z0 = float(bc[0]["point"][2]) if bc else 0.48
X = np.zeros((L + 1, n, 3), np.float32); X[0] = rng.uniform(0.8, 1.2, (n, 3)); X[0, :, 2] += z0 + 0.1 - 0.8
for t in range(1, L + 1):
    X[t] = X[t - 1]; X[t, :, 2] = np.maximum(X[t - 1, :, 2] - 0.01, z0 - 0.002 + (X[0, :, 2] - X[0, :, 2].min()))
os.makedirs("gt", exist_ok=True); os.makedirs("gt2", exist_ok=True)
json.dump({"R": np.eye(3).tolist(), "frames": L}, open("gt/meta.json", "w"))
for t in range(L + 1):
    with h5py.File(f"gt/sim_{t:010d}.h5", "w") as h:
        h["x"] = X[t]; h["F"] = (np.eye(3)[None] + 0.01 * rng.standard_normal((n, 3, 3))).reshape(n, 9).astype(np.float32)
    with h5py.File(f"gt2/sim_{t:010d}.h5", "w") as h:
        h["x"] = X[t] + 0.5; h["F"] = np.tile(np.eye(3).reshape(1, 9), (n, 1)).astype(np.float32)
def dump(name, mode):
    pred = torch.tensor(X[1:]).clone(); F = torch.eye(3).repeat(L, n, 1, 1)
    if mode == "stop":
        pred[5:] = pred[4]
    if mode == "stretch":
        F[8:, :, 0, 0] = 2.0
    if mode == "broken":
        F[10:, :100, 2, 2] = -0.5; pred[15, 0, 0] = float("nan")
    torch.save({"pred": pred, "gt": torch.tensor(X[1:]), "x0": torch.tensor(X[0]), "EXT": 1.5, "F": F.float(), "fscal": None, "tag": "0000", "t0": 0}, name)
for m in ("ok", "stop", "stretch", "broken"):
    dump(f"roll_{m}.pt", m)
for k, col in enumerate(((255, 0, 0), (0, 255, 0))):
    w = imageio.get_writer(f"v{k}.mp4", fps=30, codec="libx264", macro_block_size=1)
    for t in range(L + 1 - k):
        f = np.full((240, 320, 3), 255, np.uint8); f[t * 5:t * 5 + 40, 100:140] = col; w.append_data(f)
    w.close()
print("가짜 데이터 OK")
PY
python -u $ME/exe/allmodels_gtcheck.py --dump roll_ok.pt --cand test=$S/gt --cand train=$S/gt2 --out gtc.json; grep -q "GTDIR $S/gt$" <(python -u $ME/exe/allmodels_gtcheck.py --dump roll_ok.pt --cand test=$S/gt --cand train=$S/gt2 --out gtc.json) && echo "[gtcheck] 맞는 쪽 고름 OK"
python -u $ME/exe/allmodels_failmode.py --traj 0000 --gt_dir $S/gt --pt fake.pt --floor_cfg $W/lbpool/cfg/lego_jelly.json \
  --dump ok=roll_ok.pt --dump stop=roll_stop.pt --dump stretch=roll_stretch.pt --dump broken=roll_broken.pt --out fm_0000.json || echo SMOKE_FAIL
cp fm_0000.json fm_L000.json; python -c "import json;d=json.load(open('fm_L000.json'));d['traj']='L000';json.dump(d,open('fm_L000.json','w'))"
: > man.tsv; for m in ok stop stretch broken; do for T in L000 0000; do printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $m "라벨 $m" $T roll_$m.pt x.log 1.234 15000 15000 abc /x/ck.pt 새로 >> man.tsv; done; done
echo "OK=1" > cc.txt; echo "대조" > ec.txt
python -u $ME/exe/allmodels_report.py err --manifest man.tsv --gtdir0000 $S/gt --cmdcheck cc.txt --errcmp ec.txt --commit $C --out err.md && head -12 err.md || echo SMOKE_FAIL
python -u $ME/exe/allmodels_report.py fail --manifest man.tsv --fm fm_L000.json fm_0000.json --font /home/dkta/.fonts/NotoSansCJKkr-Regular.otf --commit $C --out fail.md --png fail.png && cat fail.md || echo SMOKE_FAIL
python -u $ME/exe/allmodels_grid.py --cell "정답 (MPM)||v0.mp4" --cell "아주 긴 모델 이름 softplus+미끄러짐 방지 다섯 궤적 (기준)|오차 12.208% · 반복 15000|v1.mp4" --cell "셋째|부제|v0.mp4" \
  --cols 2 --font /home/dkta/.fonts/NotoSansCJKkr-Regular.otf --head "궤적 0000 시험" --out grid.mp4 || echo SMOKE_FAIL
ls -la $S; rclone copy $S r2:storage/result/anchorflow/lbours/allmodels/smoke_$C -q
echo SMOKE_END
