"""ficus 다섯 궤적 적합용 충격 입력 (사용자 10-21): 공식 F_pg.pt 의 cfg 만 바꾼 출력만 최적화 입력 .pt.

풀 학습(--pool_impulse lo,hi, scene_pool._drop_init 충격 모드)과 같은 분포에서 시드 고정으로 충격을 뽑는다:
  공식 particle_impulse 힘 f 를 위쪽 축(mpm_space_vertical_upward_axis) 둘레로 θ~U[0,2π) 회전, 크기 s~U[lo,hi] 배.
  (풀 학습은 U[0,2]; 여기서는 거의 안 움직이는 표본을 피하려고 lo=0.5 -- 사용자 지시)
**정답 궤적이 아니다**: x·F·v 는 0 프레임 정지 자세(공식 F_pg 0 프레임)를 프레임 수만큼 되풀이한 자리 채움이다
(train_deform 의 롤아웃 길이가 x 프레임 수에 묶여 있어서). 그래서 이 .pt 로 낸 '롤아웃 오차' 는 아무 의미가 없다.
쓰는 것: cfg(충격), x[0], v[0], F[0] -- 초기 상태와 충격뿐.
"""
import argparse, copy, json, math, os
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="공식 F_pg.pt")
ap.add_argument("--out_root", required=True, help="<out_root>/<tag>/<tag>.pt 로 쓴다")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--seed", type=int, default=20261021)
ap.add_argument("--lo", type=float, default=0.5)
ap.add_argument("--hi", type=float, default=2.0)
ap.add_argument("--prefix", default="ficus_i")
ap.add_argument("--log", required=True, help="뽑은 방향·크기 json")
a = ap.parse_args()

d = torch.load(a.src, map_location="cpu", weights_only=False)
cfg = d["cfg"]
imps = [i for i, b in enumerate(cfg.get("boundary_conditions") or []) if b.get("type") == "particle_impulse"]
assert len(imps) == 1 and float(cfg["boundary_conditions"][imps[0]].get("start_time", 0.0)) == 0.0, imps
f = torch.tensor(cfg["boundary_conditions"][imps[0]]["force"], dtype=torch.float64)
u = torch.tensor(cfg.get("mpm_space_vertical_upward_axis", [0, 0, 1]), dtype=torch.float64)
u = u / u.norm()
T = d["x"].shape[0]
gen = torch.Generator().manual_seed(a.seed)
rec = []
for k in range(1, a.n + 1):
    th = 2.0 * math.pi * float(torch.rand(1, generator=gen))
    s = a.lo + (a.hi - a.lo) * float(torch.rand(1, generator=gen))
    fr = f * math.cos(th) + torch.cross(u, f, dim=0) * math.sin(th) + u * float(u @ f) * (1.0 - math.cos(th))
    fs = (s * fr).tolist()
    tag = f"{a.prefix}{k}"
    c2 = copy.deepcopy(cfg)
    c2["boundary_conditions"][imps[0]]["force"] = fs
    c2["imp_sample"] = dict(theta_deg=math.degrees(th), scale=s, seed=a.seed, k=k, official_force=f.tolist(),
                            note="정답 궤적 아님: x/F/v 는 0 프레임 되풀이 자리 채움")
    o = {"x": d["x"][:1].expand(T, *d["x"].shape[1:]).clone(),
         "F": d["F"][:1].expand(T, *d["F"].shape[1:]).clone(),
         "v": d["v"][:1].expand(T, *d["v"].shape[1:]).clone(),
         "sel": d["sel"], "cfg": c2, "n_full": d["n_full"], "meta": dict(d.get("meta") or {}, placeholder_frames=True),
         "gaussians_first": d.get("gaussians_first")}
    os.makedirs(f"{a.out_root}/{tag}", exist_ok=True)
    torch.save(o, f"{a.out_root}/{tag}/{tag}.pt")
    r = dict(tag=tag, theta_deg=round(math.degrees(th), 3), scale=round(s, 4), force=[round(z, 6) for z in fs])
    rec.append(r)
    print(f"[충격 표본] {tag}: 방향 θ={r['theta_deg']:.1f}° (위쪽 축 둘레, 공식 힘 {f.tolist()} 기준), 크기 x{r['scale']:.3f}, "
          f"힘 {r['force']}  -> {a.out_root}/{tag}/{tag}.pt", flush=True)
json.dump(dict(seed=a.seed, lo=a.lo, hi=a.hi, official_force=f.tolist(), up=u.tolist(), samples=rec), open(a.log, "w"), indent=1,
          ensure_ascii=False)
print("FICUS_IMP_TRAJ_OK", flush=True)
