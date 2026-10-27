"""셀 응력 모델 학습·평가 (사용자 10-27 확정 설계, doc/stress_lego.md, lib/anchorflow/stress_net.py).

모드
  stats  실제 lego 롤아웃 상태 캐시 만들기 (dump40 + fem2_lego 롤아웃 프레임) -> 실제 셀 분포 vs 생성 분포 표 (cover.txt)
  check  보존 (선·각운동량) · 회전 공변 · FCR 분해 · mom_core 색인 규약 대조 (check.txt)
  train  자기지도 학습 (무작위 셀, 정답 = 입자 적분 FCR 노드 힘). 이어 돌리기 (ck.pt), eval_every 마다 무작위 시험 셀 + lego 실제 상태 평가 -> TB·hist
  eval   체크포인트 평가 (eval.json)
  bench  추론 시간 (lego dump40 상태, 사면체 4.3 만 개 한 번 통과)

손실 (고정 상수 스케일): L = 평균_셀 Σ_n |f_n^모델 − f_n^정답|² / f_ref,c²,  f_ref,c = μ_c V_tet / node_h
  μ_c = 셀 물성 (입자 μ_p 의 부피 가중 평균 -- 입력 물성 상수이지 응답·변위가 아니다), V_tet = 격자 사면체 부피, node_h = 격자 간격 (둘 다 고정).
  = 응력을 물성 단위 μ 로 잰 무차원 힘 오차. 표본마다 응답 크기로 나누지 않는다.
평가 지표 (집합 단위 상대 오차, 분모 = 시험 집합 전체 정답 힘 크기의 합 -- 표본별로 나누지 않는다):
  무작위 시험: sqrt(Σ_c Σ_n |Δf/μ_c|² / Σ_c Σ_n |f^정답/μ_c|²)
  lego 상태: 셀 기여 (국소 힘) 상대 오차 sqrt(ΣΣ|Δf_T,n|² / ΣΣ|f_T,n|²), 전역 (노드로 합친 뒤) 상대 오차 ‖Δf‖/‖f‖
  기준 = 셀 평균 F 하나로 낸 FCR 응력 (적분점 1 개, 셀 중심), bspline2 는 같은 응력을 K 적분점에 둔 것도 함께.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import simplex as SX  # noqa: E402
from anchorflow import stress_net as SN  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["stats", "check", "train", "eval", "bench"])
ap.add_argument("--out", required=True)
ap.add_argument("--r2", default="", help="출력 R2 경로 (체크포인트·결과 파일을 저장 즉시 복사)")
ap.add_argument("--commit", default="?")
ap.add_argument("--interp", default="bary", choices=["bary", "bspline2"])
ap.add_argument("--K", type=int, default=1)
ap.add_argument("--node_h", type=float, default=0.028284)
ap.add_argument("--E", type=float, default=1e6, help="씬 E (lego)")
ap.add_argument("--nu", type=float, default=0.3)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--n_round", type=int, default=2)
ap.add_argument("--gen", default="{}", help="생성기 설정 덮어쓰기 JSON (stress_net.GEN_DEFAULTS)")
ap.add_argument("--real", default="/home/dkta/work/lbours/stress_lego/real_states.pt", help="실제 상태 캐시")
ap.add_argument("--dump40", default="/home/dkta/work/lbours/fem_rt_lego/dump40/dump/dump_t0040.pt")
ap.add_argument("--roll", default="/home/dkta/work/lbours/fem2_lego/run/roll_fem.pt")
ap.add_argument("--roll_frames", default="9,19,29,39,49,59,69,79")
ap.add_argument("--n_cover", type=int, default=16384)
ap.add_argument("--iters", type=int, default=100000)
ap.add_argument("--sched_iters", type=int, default=0, help="학습률 일정 길이 (0 이면 --iters). 이어 돌리기 시험은 같게 고정")
ap.add_argument("--batch", type=int, default=1024, help="반복당 셀 수")
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--lr_min", type=float, default=0.02, help="코사인 끝 비율")
ap.add_argument("--warm", type=int, default=1000)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval_every", type=int, default=2000)
ap.add_argument("--ckpt_every", type=int, default=1000)
ap.add_argument("--log_every", type=int, default=100)
ap.add_argument("--n_test", type=int, default=8192)
ap.add_argument("--test_seed", type=int, default=12345)
ap.add_argument("--no_real_eval", action="store_true")
ap.add_argument("--resume", default="", help="eval/bench 용 체크포인트")
a = ap.parse_args()

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.makedirs(a.out, exist_ok=True)
LAT = SX.lattice(a.node_h, SN.HZ * a.node_h, device=dev, dtype=torch.float32)
V_TET = float(torch.linalg.det(LAT.A.double()).abs()) / 6.0
MU_REF, LAM_REF = SN.CellGen.lame(a.E, a.nu)
GEN_KW = json.loads(a.gen)


def log(*s):
    print(*s, flush=True)


def r2_copy(path, bg=True):
    if not a.r2:
        return
    dst = a.r2.rstrip("/") + "/" + os.path.relpath(path, a.out)
    cmd = ["rclone", "copyto", path, dst, "-q"]
    if bg:
        threading.Thread(target=subprocess.run, args=(cmd,), daemon=False).start()
    else:
        subprocess.run(cmd)


def save_json(obj, name):
    p = os.path.join(a.out, name)
    with open(p, "w") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    r2_copy(p)
    return p


# ---------------------------------------------------------------- 실제 상태 캐시
def build_real():
    if os.path.exists(a.real):
        return torch.load(a.real, map_location="cpu", weights_only=False)
    log(f"[실제 상태] 캐시 만들기 -> {a.real}")
    d = torch.load(a.dump40, map_location="cpu", weights_only=False, mmap=True)
    N = int(d["x"].shape[0])
    vol = d["vol"]
    vol = vol.float().reshape(N) if torch.is_tensor(vol) else torch.full((N,), float(vol))
    E, nu = float(d.get("E") or a.E), float(d.get("nu") or a.nu)
    states = [dict(name="dump40 (fem_rt_lego 기준 실행 프레임 40)", x=d["x"].float().clone(), F=d["F"].float().reshape(N, 3, 3).clone())]
    log(f"  dump40: 입자 {N}, E {E:g} ν {nu}, 부피 합 {float(vol.sum()):.6g}, 질량 합 {float(d['m'].sum()):.6g}, h {d.get('h')}")
    del d
    r = torch.load(a.roll, map_location="cpu", weights_only=False, mmap=True)
    log(f"  롤아웃 덤프 {a.roll}: 키 {sorted(k for k in r if r[k] is not None)}, pred {tuple(r['pred'].shape)}, "
        f"F {tuple(r['F'].shape) if r.get('F') is not None else None} {r['F'].dtype if r.get('F') is not None else ''}")
    if r.get("F") is None:
        raise SystemExit("[오류] 롤아웃 덤프에 F 가 없다")
    if int(r["pred"].shape[1]) != N:
        raise SystemExit(f"[오류] 롤아웃 입자 수 {int(r['pred'].shape[1])} != dump40 {N} (부피를 옮길 수 없다)")
    for i in [int(s) for s in a.roll_frames.split(",") if s]:
        if i >= r["pred"].shape[0]:
            continue
        states.append(dict(name=f"fem2_lego 롤아웃 프레임 {i + 1}", x=r["pred"][i].float().clone(), F=r["F"][i].float().reshape(N, 3, 3).clone()))
    del r
    out = dict(states=states, Vp=vol, E=E, nu=nu, src=dict(dump40=a.dump40, roll=a.roll, frames=a.roll_frames))
    os.makedirs(os.path.dirname(a.real), exist_ok=True)
    torch.save(out, a.real + ".tmp")
    os.replace(a.real + ".tmp", a.real)                          # R2 복사는 드라이버 (stress_lego.sh) 가 바로 한다
    return out


def real_batch(R, s):
    mu, lam = SN.CellGen.lame(R["E"], R["nu"])
    x = s["x"].to(dev)
    N = x.shape[0]
    return SN.batch_from_particles(x, s["F"].to(dev), R["Vp"].to(dev), torch.tensor([mu], device=dev), torch.tensor([lam], device=dev), a.node_h)


# ---------------------------------------------------------------- 평가
def rel(d2, t2):
    return math.sqrt(float(d2) / max(float(t2), 1e-300))


def global_pair(fa, fb, ci):
    """두 국소 힘을 같은 노드 집합으로 전역 조립 -> (fa_g, fb_g)."""
    _ijk, F = SN.global_assemble(torch.cat([fa, fb], -1), ci)
    return F[:, :3], F[:, 3:]


@torch.no_grad()
def eval_batch(net, b, kind, K):
    ft, _ = SN.truth_forces(kind, b, LAT)
    tq, VT = net(b)
    fm = SN.quad_forces(kind, b, LAT, tq, VT, K)
    tb1, VT1 = SN.baseline_tau(b, 1)
    fb1 = SN.quad_forces(kind, b, LAT, tb1, VT1, 1)
    out = dict(ft=ft, fm=fm, fb1=fb1)
    if K > 1:
        tbk, _ = SN.baseline_tau(b, K)
        out["fbk"] = SN.quad_forces(kind, b, LAT, tbk, VT1, K)
    return out


TEST = {}


@torch.no_grad()
def eval_random(net, kind, K):
    if "b" not in TEST:
        G = SN.CellGen(LAT, a.E, a.nu, dev, **GEN_KW)
        g = torch.Generator(device=dev).manual_seed(a.test_seed)
        TEST["b"] = [G.sample(min(1024, a.n_test - i), g) for i in range(0, a.n_test, 1024)]
    acc = {}
    bins = {}
    for b, info in TEST["b"]:
        r = eval_batch(net, b, kind, K)
        _VT, _Fb, muc, _lc = SN.cell_means(b)
        st = SN.cell_stats(b, LAT, max_keep=10 ** 9)["셀 최대 |F'_p − F̄'|"].to(dev)
        w = 1.0 / muc.reshape(-1, 1, 1) ** 2
        t2 = (r["ft"] ** 2 * w).sum((1, 2))
        for k in [k for k in r if k != "ft"]:
            d2 = ((r[k] - r["ft"]) ** 2 * w).sum((1, 2))
            acc.setdefault(k, [0.0, 0.0])
            acc[k][0] += float(d2.sum()); acc[k][1] += float(t2.sum())
            # 입자 F 퍼짐 크기별 (셀 최대 |F'_p − F̄'| 구간)
            for nm, lo, hi in [("퍼짐<0.01", 0, 0.01), ("0.01~0.1", 0.01, 0.1), ("0.1~1", 0.1, 1.0), ("≥1", 1.0, 1e9)]:
                m = (st >= lo) & (st < hi)
                bins.setdefault(nm, {}).setdefault(k, [0.0, 0.0])
                bins[nm][k][0] += float(d2[m].sum()); bins[nm][k][1] += float(t2[m].sum())
    res = {k: rel(*v) for k, v in acc.items()}
    res["구간"] = {nm: {k: rel(*v) for k, v in d.items()} for nm, d in bins.items()}
    return res


REAL = {}


@torch.no_grad()
def eval_real(net, kind, K):
    if "R" not in REAL:
        REAL["R"] = build_real()
    R = REAL["R"]
    rows = []
    for s in R["states"]:
        b = real_batch(R, s)
        r = eval_batch(net, b, kind, K)
        row = dict(name=s["name"], T=b["T"], N=int(b["F"].shape[0]))
        t2 = float((r["ft"] ** 2).sum())
        for k in [k for k in r if k != "ft"]:
            row[f"셀_{k}"] = rel(((r[k] - r["ft"]) ** 2).sum(), t2)
            ga, gb = global_pair(r[k], r["ft"], b["ci"])
            row[f"전역_{k}"] = rel(((ga - gb) ** 2).sum(), (gb ** 2).sum())
        rows.append(row)
        del b, r
    keys = [k for k in rows[0] if k.startswith("셀_") or k.startswith("전역_")]
    mean = {k: float(np.mean([r_[k] for r_ in rows])) for k in keys}
    return dict(states=rows, mean=mean)


def fmt_eval(er, ee):
    s = (f"무작위 시험 힘 오차 모델 {100 * er['fm']:.3f}% | 기준(평균 F 적분점 1) {100 * er['fb1']:.3f}%"
         + (f" | 기준(평균 F 적분점 {a.K}) {100 * er['fbk']:.3f}%" if 'fbk' in er else ""))
    if ee:
        m = ee["mean"]
        s += (f"\n    lego 실제 상태 {len(ee['states'])} 개 평균: 셀 기여 모델 {100 * m['셀_fm']:.3f}% 기준 {100 * m['셀_fb1']:.3f}%"
              + (f" 기준K {100 * m['셀_fbk']:.3f}%" if '셀_fbk' in m else "")
              + f" | 전역 노드 모델 {100 * m['전역_fm']:.3f}% 기준 {100 * m['전역_fb1']:.3f}%"
              + (f" 기준K {100 * m['전역_fbk']:.3f}%" if '전역_fbk' in m else ""))
        r0 = ee["states"][0]
        s += f"\n    {r0['name']}: 셀 모델 {100 * r0['셀_fm']:.3f}% 기준 {100 * r0['셀_fb1']:.3f}% | 전역 모델 {100 * r0['전역_fm']:.3f}% 기준 {100 * r0['전역_fb1']:.3f}%"
    return s


def make_net():
    return SN.StressNet(a.K, a.interp, LAT, MU_REF, H=a.hidden, n_round=a.n_round).to(dev)


def load_ck(p):
    ck = torch.load(p, map_location=dev, weights_only=False)
    return ck


# ---------------------------------------------------------------- stats
if a.mode == "stats":
    R = build_real()
    allr = {}
    for s in tqdm(R["states"], desc="실제 상태 통계", ncols=100):
        b = real_batch(R, s)
        st = SN.cell_stats(b, LAT)
        pk = "입자 |F'_p − F̄'|"
        log(f"[실제] {s['name']}: 입자 {int(b['F'].shape[0])}, 사면체 {b['T']}, 셀 입자 수 평균 {float(st['셀 입자 수'].float().mean()):.2f} "
            f"최대 {int(st['셀 입자 수'].max())}, |F'_p − F̄'| 평균 {float(st[pk].mean()):.4f} 최대 {float(st[pk].max()):.3f}")
        for k, t in st.items():
            allr.setdefault(k, []).append(t)
    allr = {k: torch.cat(v) for k, v in allr.items()}
    G = SN.CellGen(LAT, a.E, a.nu, dev, **GEN_KW)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    allg = {}
    for i in tqdm(range(0, a.n_cover, 1024), desc="생성 셀 통계", ncols=100):
        b, _ = G.sample(min(1024, a.n_cover - i), g)
        for k, t in SN.cell_stats(b, LAT).items():
            allg.setdefault(k, []).append(t)
    allg = {k: torch.cat(v) for k, v in allg.items()}
    qs = [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
    lines = [f"[분포 비교] 실제 = lego 롤아웃 상태 {len(R['states'])} 개 ({', '.join(s['name'] for s in R['states'])}), "
             f"생성 = CellGen {a.n_cover} 셀 (설정 {json.dumps(dict(SN.GEN_DEFAULTS, **GEN_KW), ensure_ascii=False)})",
             "분위수: " + " ".join(f"{q:g}" for q in qs) + "   | 덮음 = 생성 [0.1%, 99.9%] 분위 ⊇ 실제 [0.1%, 99.9%] 분위"]
    js = {}
    for k in allr:
        r_, g_ = allr[k].double(), allg[k].double()
        r_ = r_[torch.isfinite(r_)]
        g_ = g_[torch.isfinite(g_)]
        qr = torch.quantile(r_[:2 ** 24 - 1] if r_.numel() >= 2 ** 24 else r_, torch.tensor(qs, dtype=torch.float64)).tolist()
        qg = torch.quantile(g_[:2 ** 24 - 1] if g_.numel() >= 2 ** 24 else g_, torch.tensor(qs, dtype=torch.float64)).tolist()
        cov = (qg[1] <= qr[1] + 1e-12) and (qg[7] >= qr[7] - 1e-12)
        lines.append(f"{k:28s} 실제 " + " ".join(f"{v:10.4g}" for v in qr) + f"   (표본 {r_.numel()})")
        lines.append(f"{'':28s} 생성 " + " ".join(f"{v:10.4g}" for v in qg) + f"   (표본 {g_.numel()})  {'덮음' if cov else '** 못 덮음 **'}")
        js[k] = dict(real=qr, gen=qg, covered=bool(cov))
    txt = "\n".join(lines)
    log(txt)
    open(os.path.join(a.out, "cover.txt"), "w").write(txt + "\n")
    r2_copy(os.path.join(a.out, "cover.txt"))
    save_json(dict(q=qs, stats=js, gen=dict(SN.GEN_DEFAULTS, **GEN_KW)), "cover.json")
    log("STRESS_STATS_DONE")
    sys.exit(0)


# ---------------------------------------------------------------- check
if a.mode == "check":
    torch.manual_seed(0)
    L = []

    def say(s):
        log(s); L.append(s)
    R = build_real()
    s0 = R["states"][0]
    # (1) FCR 분해 = deform_model.kirchhoff_tau (씬 cfg) 그대로
    from anchorflow import deform_model as DM
    Fr = s0["F"][:200000].to(dev).double()
    mu, lam = SN.CellGen.lame(R["E"], R["nu"])
    t_ref = DM.kirchhoff_tau(Fr, {"E": R["E"], "nu": R["nu"], "material": "jelly"})
    t_my = SN.tau_fcr(Fr, torch.full((Fr.shape[0],), mu, device=dev), torch.full((Fr.shape[0],), lam, device=dev))
    say(f"[FCR 분해] 입자별 μ·λ 합성 vs kirchhoff_tau(씬 E {R['E']:g}, ν {R['nu']}) (dump40 입자 20 만, float64): 상대차 "
        f"{float((t_my - t_ref).norm() / t_ref.norm()):.3e}, 대칭성 ‖τ−τᵀ‖/‖τ‖ {float((t_ref - t_ref.transpose(1, 2)).norm() / t_ref.norm()):.3e}")
    # (2) 보간 항등식 (무작위 점, float64)
    LAT64 = SX.lattice(a.node_h, SN.HZ * a.node_h, device=dev, dtype=torch.float64)
    G32 = SN.CellGen(LAT, a.E, a.nu, dev, **GEN_KW)
    gg = torch.Generator(device=dev).manual_seed(1)
    perm = G32.perms[torch.randint(0, 6, (20000,), generator=gg, device=dev)]
    lamb = SN._dirichlet1(gg, 20000, dev).double()
    fpt = SN.lam_to_y(lamb, perm)
    for kind in ("bary", "bspline2"):
        off, gr, w = SN.interp_local(kind, fpt, perm, LAT64)
        xn = off.double() @ LAT64.A.T                                         # 노드 세계 위치 (ci = 0, lo = 0)
        xp = fpt @ LAT64.A.T
        S1 = float((w.sum(1) - 1).abs().max())
        S2 = float(gr.sum(1).norm(dim=-1).max() * a.node_h)
        S3 = float((torch.einsum("pni,pnj->pij", xn, gr) - torch.eye(3, device=dev, dtype=torch.float64)).abs().max())
        S4 = float(((w.unsqueeze(-1) * xn).sum(1) - xp).norm(dim=-1).max() / a.node_h)
        say(f"[보간 {kind}] 무작위 점 2 만 (float64): |Σw−1| 최대 {S1:.2e}, |Σ∇w|·h 최대 {S2:.2e}, |Σ x_n⊗∇w_n − I| 최대 {S3:.2e}, "
            f"|Σ w x_n − x|/h 최대 {S4:.2e}")
    # (3) 보존 + (4) 회전 공변: 무작위 셀, 망 출력 층을 무작위로 (0 초기화면 보정이 꺼져 있다)
    for kind, K in (("bary", 1), ("bary", 4), ("bspline2", 1), ("bspline2", 4)):
        net = SN.StressNet(K, kind, LAT64, MU_REF, H=64, n_round=2).to(dev).double()
        torch.nn.init.normal_(net.out[-1].weight, std=0.3)
        torch.nn.init.normal_(net.out[-1].bias, std=0.3)
        b, _ = G32.sample(2048, gg)
        b = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in b.items()}
        with torch.no_grad():
            tq, VT = net(b)
            fm = SN.quad_forces(kind, b, LAT64, tq, VT, K)
            ft, _ = SN.truth_forces(kind, b, LAT64)
            xn = SN.local_offsets(dev).double() @ LAT64.A.T                    # [64,3]
            res = {}
            skew = torch.randn_like(tq) * tq.abs().mean()
            fs = SN.quad_forces(kind, b, LAT64, tq + (skew - skew.transpose(-1, -2)), VT, K)
            for nm, f in (("모델", fm), ("정답", ft), ("모델+반대칭 τ (대조)", fs)):
                xc = (xn.unsqueeze(0) * (f.norm(dim=-1, keepdim=True) > 0)).sum(1) / (f.norm(dim=-1) > 0).sum(1, keepdim=True).clamp_min(1)
                r_ = xn.unsqueeze(0) - xc.unsqueeze(1)
                lin = f.sum(1).norm(dim=-1) / f.norm(dim=-1).sum(1).clamp_min(1e-300)
                angm = torch.cross(r_, f, dim=-1).sum(1).norm(dim=-1) / (r_.norm(dim=-1) * f.norm(dim=-1)).sum(1).clamp_min(1e-300)
                res[nm] = (float(lin.max()), float(angm.max()))
            # 회전 공변
            Q = SN._rand_rot(gg, b["T"], dev).double()
            b2 = dict(b); b2["F"] = Q[b["seg"]] @ b["F"]
            tq2, _ = net(b2)
            cov = float(((tq2 - Q.unsqueeze(1) @ tq @ Q.transpose(1, 2).unsqueeze(1)).flatten(1).norm(dim=1) / tq.flatten(1).norm(dim=1).clamp_min(1e-300)).max())
            fm2 = SN.quad_forces(kind, b2, LAT64, tq2, VT, K)
            # 힘도 회전: f(QF) 의 셀 힘 = ? (τ -> QτQᵀ 이므로 f_n -> −Σ V Qτ Qᵀ ∇w -- 노드 힘 자체는 회전 공변이 아니다 (격자 고정); 응력만 확인)
            net32 = SN.StressNet(K, kind, LAT, MU_REF, H=64, n_round=2).to(dev)
            net32.load_state_dict({k: v.float() for k, v in net.state_dict().items()})
            b32 = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in b.items()}
            b32_2 = dict(b32); b32_2["F"] = Q.float()[b32["seg"]] @ b32["F"]
            t1, _ = net32(b32); t2, _ = net32(b32_2)
            cov32 = float(((t2 - Q.float().unsqueeze(1) @ t1 @ Q.float().transpose(1, 2).unsqueeze(1)).flatten(1).norm(dim=1)
                           / t1.flatten(1).norm(dim=1).clamp_min(1e-30)).max())
            sym = float((tq - tq.transpose(-1, -2)).abs().max() / tq.abs().max())
        say(f"[보존 {kind} K={K}] 무작위 셀 2048 개 (float64, 출력 층 무작위): 셀별 |Σ_n f_n|/Σ_n|f_n| 최대 — 모델 {res['모델'][0]:.2e}, 정답 {res['정답'][0]:.2e}; "
            f"|Σ_n (x_n−x̄)×f_n|/Σ|x_n−x̄||f_n| 최대 — 모델 {res['모델'][1]:.2e}, 정답 {res['정답'][1]:.2e}, "
            f"반대칭 τ 를 더한 대조 {res['모델+반대칭 τ (대조)'][1]:.2e} (τ 대칭성 {sym:.1e})")
        say(f"[회전 공변 {kind} K={K}] F_p -> Q F_p (셀마다 무작위 Q ∈ SO(3)): max_셀 ‖τ_q(QF) − Q τ_q(F) Qᵀ‖/‖τ_q‖ = {cov:.2e} (float64), {cov32:.2e} (float32)")
        del net, b, b2
    # (5) mom_core 색인 규약 대조 (lego 실제 상태, bary)
    from anchorflow import mom_core as MC
    x = s0["x"].to(dev)
    S = MC.frame(x, a.node_h)
    b = real_batch(R, s0)
    say(f"[mom_core 대조] dump40: mom_core 사면체 {S.T}, 이쪽 셀 {b['T']}, 입자 셀 번호 같음 {bool((S.pinv == b['seg']).all())}")
    nnl = [int(S.nn[k]) for k in range(3)]
    gv = b["ci"].unsqueeze(1) + SN.tet_verts(b["perm"])                       # [T,4,3]
    flat = (gv[..., 0] * nnl[1] + gv[..., 1]) * nnl[2] + gv[..., 2]
    say(f"  사면체 꼭짓점 (사슬 순서) 같음: {bool((S.uniq[S.tets] == flat).all())}, ∇λ 최대차 "
        f"{float((S.gl - SN.bary_grads(LAT, b['perm'])).abs().max() / S.gl.abs().max()):.2e} (상대)")
    ft, taup = SN.truth_forces("bary", b, LAT)
    VT = SN.seg_sum(b["Vp"], b["seg"], b["T"])
    tauT = SN.seg_sum(b["Vp"][:, None, None] * taup, b["seg"], b["T"]) / VT[:, None, None]
    f_mc = MC.node_force(S, VT, tauT)                                          # bary 정확: 입자 τ 의 부피 평균
    ijk, fg = SN.global_assemble(ft, b["ci"])
    flat_g = (ijk[:, 0] * nnl[1] + ijk[:, 1]) * nnl[2] + ijk[:, 2]
    pos = torch.searchsorted(S.uniq, flat_g)
    ok = bool((S.uniq[pos.clamp(max=S.uniq.numel() - 1)] == flat_g).all())
    f_mc_on = f_mc[pos]
    say(f"  노드 힘: mom_core.node_force(VT, 입자 τ 부피평균) vs 이쪽 입자 적분 정답 전역 조립 -- 노드 집합 일치 {ok} ({fg.shape[0]} / {S.M}), "
        f"상대차 {float((f_mc_on - fg).norm() / fg.norm()):.2e}")
    net = make_net()
    torch.nn.init.normal_(net.out[-1].weight, std=0.1)
    if a.interp == "bary":
        ad = SN.MomStressAdapter(net)
        mu, lam = SN.CellGen.lame(R["E"], R["nu"])
        tT = ad.tau_T(S, x, b["F"], b["Vp"], mu, lam)
        f_ad = MC.node_force(S, VT, tT)
        with torch.no_grad():
            tq, VT2 = net(b)
            fl = SN.quad_forces("bary", b, LAT, tq, VT2, a.K)
        ijk2, fg2 = SN.global_assemble(fl, b["ci"])
        flat2 = (ijk2[:, 0] * nnl[1] + ijk2[:, 1]) * nnl[2] + ijk2[:, 2]
        say(f"  어댑터: MomStressAdapter.tau_T -> mom_core.node_force vs 이쪽 적분점 조립 (망 출력 층 무작위) 상대차 "
            f"{float((f_ad[torch.searchsorted(S.uniq, flat2)] - fg2).norm() / fg2.norm()):.2e}")
    open(os.path.join(a.out, "check.txt"), "w").write("\n".join(L) + "\n")
    r2_copy(os.path.join(a.out, "check.txt"))
    log("STRESS_CHECK_DONE")
    sys.exit(0)


# ---------------------------------------------------------------- bench
if a.mode == "bench":
    net = make_net()
    if a.resume:
        net.load_state_dict(load_ck(a.resume)["net"])
    net.eval()
    R = build_real()
    s0 = R["states"][0]
    gpu = torch.cuda.get_device_name()
    nv = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout
    res = dict(gpu=gpu, nvidia_smi=nv, interp=a.interp, K=a.K)

    def tm(fn, n=20, w=5):
        for _ in range(w):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts)) * 1e3, float(np.min(ts)) * 1e3
    x = s0["x"].to(dev)
    b = real_batch(R, s0)
    res["T"], res["N"] = b["T"], int(b["F"].shape[0])
    with torch.no_grad():
        res["격자·셀 묶기 ms"] = tm(lambda: real_batch(R, s0))
        res["모델 응력 (회전 정규화+망+τ_FCR(F̄') 기저+되돌림) ms"] = tm(lambda: net(b))
        def full():
            tq, VT = net(b)
            fl = SN.quad_forces(a.interp, b, LAT, tq, VT, a.K)
            return SN.global_assemble(fl, b["ci"])
        res["모델 응력 + 노드 힘 조립 ms"] = tm(full)
        def base():
            tb, VT = SN.baseline_tau(b, 1)
            return SN.global_assemble(SN.quad_forces(a.interp, b, LAT, tb, VT, 1), b["ci"])
        res["기준 (평균 F 적분점 1) 응력 + 조립 ms"] = tm(base)
        res["정답 (입자 FCR 적분) 응력 + 조립 ms"] = tm(lambda: SN.global_assemble(SN.truth_forces(a.interp, b, LAT)[0], b["ci"]))
        torch.cuda.reset_peak_memory_stats()
        full(); torch.cuda.synchronize()
        res["모델 최대 메모리 GB"] = torch.cuda.max_memory_allocated() / 1e9
    for k, v in res.items():
        log(f"[추론 시간] {k}: {v}")
    save_json(res, "bench.json")
    log("STRESS_BENCH_DONE")
    sys.exit(0)


# ---------------------------------------------------------------- eval
if a.mode == "eval":
    net = make_net()
    ck = load_ck(a.resume)
    net.load_state_dict(ck["net"])
    net.eval()
    er = eval_random(net, a.interp, a.K)
    ee = eval_real(net, a.interp, a.K)
    log(f"[평가] 체크포인트 {a.resume} (반복 {ck['it']})\n  " + fmt_eval(er, ee))
    save_json(dict(it=ck["it"], random=er, real=ee, ckpt=a.resume, commit=a.commit), "eval.json")
    log("STRESS_EVAL_DONE")
    sys.exit(0)


# ---------------------------------------------------------------- train
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

net = make_net()
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
GEN = SN.CellGen(LAT, a.E, a.nu, dev, **GEN_KW)
gen = torch.Generator(device=dev).manual_seed(a.seed)
SI = a.sched_iters or a.iters
it0, hist, best = 0, [], None
ckp = os.path.join(a.out, "ck.pt")
if os.path.exists(ckp):
    ck = load_ck(ckp)
    net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
    gen.set_state(ck["gen"].cpu() if torch.is_tensor(ck["gen"]) else ck["gen"])
    torch.set_rng_state(ck["cpu_rng"].cpu())
    it0, hist, best = int(ck["it"]), ck["hist"], ck.get("best")
    log(f"[이어 돈다] {ckp} 반복 {it0} (평가 기록 {len(hist)}, 최저 {best})")
npar = sum(p.numel() for p in net.parameters())
log(f"[설정] 커밋 {a.commit} 보간 {a.interp} K {a.K} 망 파라미터 {npar} (폭 {a.hidden}, 왕복 {a.n_round}) 반복 {a.iters} (일정 {SI}) 셀/반복 {a.batch} "
    f"lr {a.lr} node_h {a.node_h} V_tet {V_TET:.4e} μ_ref {MU_REF:.6g} 생성기 {json.dumps(dict(SN.GEN_DEFAULTS, **GEN_KW), ensure_ascii=False)}")
TB = SummaryWriter(os.path.join(a.out, "tb"), purge_step=it0 if it0 > 0 else None)


def lr_at(it):
    w = min(1.0, (it + 1) / max(1, a.warm))
    c = 0.5 * (1 + math.cos(math.pi * min(it, SI) / SI))
    return a.lr * w * (a.lr_min + (1 - a.lr_min) * c)


def save_ck(it, name="ck.pt"):
    p = os.path.join(a.out, name)
    torch.save(dict(it=it, net=net.state_dict(), opt=opt.state_dict(), gen=gen.get_state(), cpu_rng=torch.get_rng_state(), hist=hist, best=best,
                    args=vars(a)), p + ".tmp")
    os.replace(p + ".tmp", p)
    r2_copy(p)


def do_eval(it):
    global best
    net.eval()
    t0 = time.time()
    er = eval_random(net, a.interp, a.K)
    ee = None if a.no_real_eval else eval_real(net, a.interp, a.K)
    net.train()
    rec = dict(it=it, random=er, real=ee, sec=time.time() - t0)
    hist.append(rec)
    TB.add_scalar("평가/무작위_힘오차_모델", er["fm"], it)
    TB.add_scalar("평가/무작위_힘오차_기준_평균F_적분점1", er["fb1"], it)
    if "fbk" in er:
        TB.add_scalar(f"평가/무작위_힘오차_기준_평균F_적분점{a.K}", er["fbk"], it)
    for nm, d in er["구간"].items():
        TB.add_scalar(f"평가구간/무작위_{nm}_모델", d["fm"], it)
    if ee:
        m = ee["mean"]
        TB.add_scalar("평가/lego_셀힘오차_모델", m["셀_fm"], it)
        TB.add_scalar("평가/lego_셀힘오차_기준", m["셀_fb1"], it)
        TB.add_scalar("평가/lego_전역힘오차_모델", m["전역_fm"], it)
        TB.add_scalar("평가/lego_전역힘오차_기준", m["전역_fb1"], it)
        TB.add_scalar("평가/lego_dump40_전역힘오차_모델", ee["states"][0]["전역_fm"], it)
    TB.flush()
    log(f"[평가] 반복 {it} ({rec['sec']:.1f} s)\n    " + fmt_eval(er, ee))
    if best is None or er["fm"] < best["fm"]:
        best = dict(it=it, fm=er["fm"])
        save_ck(it, "ck_best.pt")
    save_json(dict(hist=hist, best=best, commit=a.commit, args=vars(a)), "hist.json")


net.train()
pbar = tqdm(range(it0, a.iters), desc=f"응력 모델 {a.interp} K{a.K}", ncols=110, initial=it0, total=a.iters)
t_last, run_l = time.time(), []
for it in pbar:
    for g_ in opt.param_groups:
        g_["lr"] = lr_at(it)
    b, info = GEN.sample(a.batch, gen)
    with torch.no_grad():
        ft, _ = SN.truth_forces(a.interp, b, LAT)
    tq, VT = net(b)
    fm = SN.quad_forces(a.interp, b, LAT, tq, VT, a.K)
    _VT, _Fb, muc, _lc = SN.cell_means(b)
    fref = muc * V_TET / a.node_h
    loss = (((fm - ft) ** 2).sum((1, 2)) / fref ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
    opt.step()
    run_l.append(float(loss))
    if not math.isfinite(run_l[-1]):
        raise SystemExit(f"[오류] 손실 비유한 반복 {it}")
    k = it + 1
    if k % a.log_every == 0:
        lm = float(np.mean(run_l)); run_l = []
        TB.add_scalar("학습/손실", lm, k)
        TB.add_scalar("학습/학습률", lr_at(it), k)
        TB.add_scalar("학습/기울기노름", float(gn), k)
        TB.add_scalar("학습/반복당_입자수", int(b["F"].shape[0]), k)
        pbar.set_postfix_str(f"손실 {lm:.3e}")
        if k % (a.log_every * 10) == 0:
            log(f"\n[반복 {k}] 손실 {lm:.4e} lr {lr_at(it):.2e} |g| {float(gn):.3e} 입자 {int(b['F'].shape[0])} {(time.time() - t_last) / (a.log_every * 10):.3f} s/반복")
            t_last = time.time()
    if k % a.ckpt_every == 0 or k == a.iters:
        save_ck(k)
    if k % a.eval_every == 0 or k == a.iters:
        do_eval(k)
        save_ck(k)
TB.close()
log(f"[끝] 반복 {a.iters}, 최저 {best}")
log("STRESS_TRAIN_DONE")
