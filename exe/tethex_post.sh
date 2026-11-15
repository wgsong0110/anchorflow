#!/bin/bash
# 절 CM180: 혼합 판 후처리 (인스턴스): pred.pt 의 입자 F det <= 0 수 (프레임별) -> flips.json, 3DGS 렌더 (lb_roll_to_h5 -> lb_render) -> vj/fem_lego.mp4, R2 lbours/tethex/<판> 로 올림.
# 사용: bash exe/tethex_post.sh <카드> <판>   (코드 디렉토리에서)
export PATH=/home/dkta/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=4
G=$1; n=$2; PY=/home/dkta/.conda/envs/af/bin/python
O=/home/dkta/work/tethex/$n; R=r2:storage/result/anchorflow/lbours/tethex/$n; W=/home/dkta/work
export TMPDIR=/dev/shm/dkta/tmp XDG_CACHE_HOME=/dev/shm/dkta/cache
$PY - $O/pred.pt $O/flips.json <<'PYE'
import sys, json, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False); F = d["F"]
out = []
for t in range(F.shape[0]):
    det = torch.linalg.det(F[t].double())
    out.append(dict(f=t, n_det_le0=int((det <= 0).sum()), n_nonfinite=int((~torch.isfinite(det)).sum()), det_min=float(det[torch.isfinite(det)].min())))
json.dump(dict(per_frame=out, first_flip=next((o["f"] for o in out if o["n_det_le0"] > 0), None), max_flip=max(o["n_det_le0"] for o in out), n_particles=int(F.shape[1])), open(sys.argv[2], "w"), indent=1)
print("flips", json.load(open(sys.argv[2]))["first_flip"], json.load(open(sys.argv[2]))["max_flip"])
PYE
HD=/dev/shm/dkta/tethex_h5/$n; mkdir -p $HD $O/vj
$PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $W/lbdata/test_low/L000 --out $HD/h5 > $O/h5.log 2>&1
CUDA_VISIBLE_DEVICES=$G $PY -u exe/lb_render.py --sim $HD/h5 --config $W/lbdata/config.json --shape lego --out $O/vj/fem_lego.mp4 --frame_label 0 --caption " " < /dev/null > $O/vj/r.log 2>&1
rm -rf $HD
rclone copy $O/vj $R/vj -q; rclone copyto $O/flips.json $R/flips.json -q; rclone copyto $O/h5.log $R/h5.log -q
echo "[post] $n $(ls -la $O/vj | tail -2 | tr '\n' ' ')"
