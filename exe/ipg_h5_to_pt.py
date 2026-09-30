"""i-PG 실행 결과(h5 프레임)를 우리 궤적 .pt 형식으로 바꾼다. 멱등이다.

기존 교사와 **같은 입자 부분표본**(`sel`)을 쓴다 -- 그래야 옛 교사와 행이 대응해
수치를 나란히 놓을 수 있다. 손잡이 정보도 같은 시나리오이므로 기준 궤적에서
그대로 가져온다. `cfg` 는 i-PG 가 실제로 돌린 config 를 넣는다 (n_grid 가 다르다).

  python exe/ipg_h5_to_pt.py --h5dir W/ipg/out_100 --ref W/traj_h2/...pt \
      --cfg W/ipg/cfg_ng100.json --out W/traj_ipg/mic_clayC_ipg100_s400706.pt
"""
import argparse, glob, json, os
import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--h5dir", required=True)
ap.add_argument("--ref", required=True, help="기준 궤적 (sel·손잡이·tag 를 물려받는다)")
ap.add_argument("--cfg", required=True, help="i-PG 가 돌린 config json")
ap.add_argument("--out", required=True)
a = ap.parse_args()

if os.path.exists(a.out):
    print(f"[건너뜀] 이미 있다: {a.out}")
    raise SystemExit(0)


def _load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def rd(p, k):
    with h5py.File(p, "r") as f:
        v = np.array(f[k])
    # PG 는 (3,N) 로 저장하는 경우가 있다
    return v.T if v.shape[0] in (3, 9) and v.shape[0] != v.shape[-1] else v


ref = _load(a.ref)
sel = ref["sel"]
files = sorted(glob.glob(os.path.join(a.h5dir, "**", "*.h5"), recursive=True))
assert files, f"h5 가 없다: {a.h5dir}"
X0 = rd(files[0], "x")
assert X0.shape[0] == int(ref["n_full"]), (
    f"입자 집합이 기준과 다르다: {X0.shape[0]} vs {int(ref['n_full'])} -- "
    "채움 설정(particle_filling)이 같은지 확인할 것")

xs, vs, fs = [], [], []
for p in files:
    xs.append(torch.from_numpy(rd(p, "x")).float()[sel])
    try:
        vs.append(torch.from_numpy(rd(p, "v")).float()[sel])
    except Exception:
        vs.append(torch.zeros_like(xs[-1]))
    try:
        fs.append(torch.from_numpy(rd(p, "f_tensor")).float()
                  .reshape(-1, 3, 3)[sel])
    except Exception as e:
        print(f"  [경고] F 를 못 읽어 항등으로 둔다: {e}")
        fs.append(torch.eye(3).repeat(sel.numel(), 1, 1))
X, V, Fm = torch.stack(xs), torch.stack(vs), torch.stack(fs)
cfg = json.load(open(a.cfg))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
torch.save({"x": X.half(), "v": V.half(), "F": Fm.half(), "sel": sel,
            "cfg": cfg, "seed": int(ref.get("seed", 0)),
            "tag": os.path.basename(a.out)[:-3],
            "ctrl_pos": ref["ctrl_pos"], "ctrl_vel": ref["ctrl_vel"],
            "ctrl_id": ref["ctrl_id"], "ctrl_R": ref["ctrl_R"],
            "n_full": int(X0.shape[0])}, a.out)
print(f"[저장] {a.out}  {X.shape[0]}프레임 x {sel.numel()}입자  "
      f"n_grid {cfg['n_grid']}  {os.path.getsize(a.out) / 1e6:.0f}MB")
