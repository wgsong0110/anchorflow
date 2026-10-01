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
ap.add_argument("--keep", type=int, default=0,
                help="저장할 입자 수. 0(기본) 이면 **전체**를 저장한다. 옛 교사는 "
                     "용량 때문에 20000 개(8%) 만 저장했는데, 학습의 매 스텝 "
                     "재추출이 그 8% 안에서만 일어나 나머지 92% 를 영원히 못 봤다. "
                     "전체를 저장하면 그 제약이 사라진다 (한 궤적 약 460MB)")
ap.add_argument("--ref_sel", action="store_true",
                help="기준 궤적의 sel 을 그대로 써서 옛 교사와 행을 맞춘다 "
                     "(수치를 직접 나란히 놓고 싶을 때만)")
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
files = sorted(glob.glob(os.path.join(a.h5dir, "**", "*.h5"), recursive=True))
assert files, f"h5 가 없다: {a.h5dir}"
X0 = rd(files[0], "x")
assert X0.shape[0] == int(ref["n_full"]), (
    f"입자 집합이 기준과 다르다: {X0.shape[0]} vs {int(ref['n_full'])} -- "
    "채움 설정(particle_filling)이 같은지 확인할 것")
if a.ref_sel:
    sel = ref["sel"]
elif a.keep > 0 and a.keep < X0.shape[0]:
    g = torch.Generator().manual_seed(1234 + int(ref.get("seed", 0)))
    sel = torch.randperm(X0.shape[0], generator=g)[:a.keep].sort().values
else:
    sel = torch.arange(X0.shape[0])          # 전체
print(f"저장 입자 {sel.numel()} / 전체 {X0.shape[0]}")

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
# PG/i-PG 는 **프레임 0 의 f_tensor 를 채우지 않는다** (전부 0 -> det=0).
# 정지 자세의 변형구배는 항등이므로 바로잡는다. 그대로 두면 t=0 에서 출발하는
# 창의 탄성 에너지가 뜻 없는 값이 된다 (Psi 가 sigma=0 에서 정의되지 않는다).
_d0 = torch.linalg.det(Fm[0].double())
if float(_d0.abs().max()) < 1e-6:
    Fm[0] = torch.eye(3).expand_as(Fm[0]).clone()
    print(f"[보정] 프레임 0 의 F 가 전부 0 이라 항등으로 채웠다 "
          f"({Fm.shape[1]} 입자)")
cfg = json.load(open(a.cfg))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
# 손잡이 위치는 **이 궤적 자신의** 것을 쓴다. 기준 궤적 것을 그대로 넣으면
# 렌더의 손잡이 원이 남의 경로를 그려 비교가 어긋난다 (실제로 그랬다).
_out = {"x": X.half(), "v": V.half(), "F": Fm.half(), "sel": sel,
        "cfg": cfg, "seed": int(ref.get("seed", 0)),
        "tag": os.path.basename(a.out)[:-3],
        "n_full": int(X0.shape[0])}
if "ctrl_id" in ref:                     # 손잡이 없는 장면도 받는다
    _hid = ref["ctrl_id"][0] if ref["ctrl_id"].dim() == 2 else ref["ctrl_id"]
    _hid = _hid.reshape(-1).long()
    _out["ctrl_pos"] = torch.stack(
        [torch.from_numpy(rd(p_, "x")).float()[_hid] for p_ in files])
    _out["ctrl_vel"] = ref["ctrl_vel"]
    _out["ctrl_id"] = ref["ctrl_id"]
    _out["ctrl_R"] = ref["ctrl_R"]
else:
    print("[손잡이] 기준 궤적에 없다 -- 손잡이 없이 저장한다")
torch.save(_out, a.out)
print(f"[저장] {a.out}  {X.shape[0]}프레임 x {sel.numel()}입자  "
      f"n_grid {cfg['n_grid']}  {os.path.getsize(a.out) / 1e6:.0f}MB")
