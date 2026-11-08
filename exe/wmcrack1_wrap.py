"""절 CM59 묶음 wmcrack1 감싸개 (수박이 녹지 않고 첫 충돌에 깨지게): 실행기를 고치지 않고 모듈 함수를 감싼 뒤 exe/ghost6_wrap.py 로 넘김.

사용: python exe/wmcrack1_wrap.py [--wc1sv 1] [--wc1k 1] <ghost6_wrap 인자...> <실행기.py> <실행기 인자...>
  --wc1sv 1  진단 (결과 불변): --wsh_sv 프레임마다 입자 (원래 순서) 탄성 F 의 특이값 σ1..3 · det F -> <out>/wsh/fs_<프레임>.npz (float32).
             층별·입자별 RM 갈래 (압축 꼭짓점·인장 꼭짓점·타원)·응집력 p0 를 정답 (GF h5 f_tensor + gt_jp) 과 같은 식으로 보려고 (exe/wmcrack1_prof.py).
  --wc1k 1   후보 K (접촉 CFL, 정의 변경; 옛 실행기 split_step·새 실행기 split_step 둘 다, 바닥 없는 씬 (ficus·bread) 은 K_req 0 = 그대로): 쪼개기 (split_step) 첫 시도의 걸음 수 K0 를 1 대신
             K_req = ⌈ max_{p ∈ C} (−v_n,p) h / Δx ⌉ 이상의 가장 작은 2 의 거듭제곱으로 (상한 = 기존 쪼개기 버퍼 2^--ksplit).
             C = 프레임 시작 속도로 이 프레임 안에 바닥 장벽 띠 (d ≤ d̂) 에 들어가는 입자 (d_p + v_n,p h ≤ d̂, v_n,p < 0),
             v_n = 바닥 법선 속도, Δx = 노드 간격 (|A[:,0]|), d̂ = 장벽 두께 (0.5 x 정답 격자 dx, 실행기 WBAR 와 같은 식).
             -- '한 걸음에 접촉 입자가 노드 한 칸 넘게 움직이지 않음' (실행기 --adaptK 1 과 같은 CFL 1 정의를 접촉 입자에만). 새 조절 값·클램프 없음.
             접촉이 없는 프레임은 K_req = 0 -> 기존 쪼개기 그대로 (K 1 부터). 무너짐 다시 하기 (--wrx_col) 는 K0 부터 2^--ksplit 까지.
  환경 AF_WC1_LOG=0 이면 wc1.jsonl 안 씀 (기본 씀; 프레임마다 K_req·K0·C 크기, 결과 불변).
"""
import json
import math
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
CAND = {}
while argv and argv[0].startswith("--wc1"):
    CAND[argv[0][2:]] = argv[1]
    del argv[:2]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)


def _arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def err(where, e_):
    import traceback
    with open(os.path.join(out, "wc1_err.txt"), "a") as f:
        f.write(f"[{where}] {e_!r}\n{traceback.format_exc()}\n")


if int(float(CAND.get("wc1sv", 0))):
    import numpy as np
    import torch
    from anchorflow import fem_wmshape as WSHM
    o_frame = WSHM.Rec.frame

    def frame(s, fi):
        r = o_frame(s, fi)
        t = fi + 1
        try:
            if t in s.sv and int(s.S.IP[13]) == 0:
                S = s.S
                Fo = s._orig(S.F.reshape(S.F.shape[0], -1)).reshape(-1, 3, 3)
                sv = torch.empty(Fo.shape[0], 3, device=Fo.device, dtype=torch.float32)
                dt_ = torch.empty(Fo.shape[0], device=Fo.device, dtype=torch.float32)
                for i in range(0, Fo.shape[0], 1 << 20):
                    Fi = Fo[i:i + (1 << 20)].double()
                    ok = torch.isfinite(Fi).flatten(1).all(1)
                    Fz = torch.where(ok[:, None, None], Fi, torch.eye(3, device=Fi.device, dtype=Fi.dtype).expand_as(Fi))
                    sv[i:i + Fi.shape[0]] = torch.where(ok[:, None], torch.linalg.svdvals(Fz), torch.full_like(Fz[:, 0], float("nan"))).float()
                    dt_[i:i + Fi.shape[0]] = torch.where(ok, torch.linalg.det(Fz), torch.full_like(Fz[:, 0, 0], float("nan"))).float()
                np.savez_compressed(os.path.join(s.out, "wsh", f"fs_{t:03d}.npz"), sv=sv.cpu().numpy(), det=dt_.cpu().numpy())
        except Exception as e_:
            err(f"wc1sv f{t}", e_)
        return r
    WSHM.Rec.frame = frame
    print("[절 CM59] --wc1sv: wsh_sv 프레임에 입자 탄성 F 특이값·det 저장 (결과 불변)", flush=True)

if int(float(CAND.get("wc1k", 0))):
    import torch
    SIM = next((q for q in argv if q.endswith(".py")), "")
    NEW = os.path.basename(SIM) == "fem_an_rt_sim.py"                                   # 새 실행기 (레고·wolf·ficus·bread) / 옛 실행기 (수박)
    _d = torch.load(_arg("--data"), map_location="cpu", weights_only=False)
    _cfg = _d["cfg"]
    del _d
    DXG = float(_cfg.get("grid_lim", 2.0)) / int(_cfg.get("n_grid", 100))
    FLR = _arg("--floor", "none" if NEW else "vel")
    if FLR == "vel":
        fp = _arg("--floor_plane", "")
        if fp:
            _q = [float(q) for q in fp.split(",")]
            PT, NV = _q[:3], _q[3:]
        else:
            pl = [b for b in _cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
            PT, NV = [float(q) for q in pl["point"]], [float(q) for q in pl["normal"]]
        NR = torch.tensor(NV, dtype=torch.float64); NR = NR / NR.norm()
        PN = float(NR @ torch.tensor(PT, dtype=torch.float64))
        # 장벽 두께 d̂ = 0.5 x 정답 격자 dx (x --bar_dhmul, 옛 실행기 WBAR DH · 새 실행기 FL DH 와 같은 식); 장벽 없는 바닥은 0
        DH = 0.5 * DXG * float(_arg("--bar_dhmul", 1.0)) if "--fl_bar" in argv else 0.0
    else:
        NR = None; PN = 0.0; DH = 0.0
    WLOG = os.environ.get("AF_WC1_LOG", "1") == "1"
    ST = dict(fi=0)

    def k_req(s, h):
        """접촉 CFL K_req (실수) 와 C 크기 (바닥 없는 씬은 0)."""
        if NR is None:
            return 0.0, 0
        x = s.x.reshape(-1, 3).double(); v = s.v.reshape(-1, 3).double()
        n = NR.to(x.device)
        d = x @ n - PN
        vn = v @ n
        ok = torch.isfinite(d) & torch.isfinite(vn)
        C = ok & (vn < 0) & (d + vn * h <= DH)
        nC = int(C.sum())
        if nC == 0:
            return 0.0, 0
        dx = float(s.A.reshape(3, 3)[:, 0].double().norm())
        return float((-vn[C]).max()) * h / dx, nC

    def depth0(s, h):
        try:
            r, nC = k_req(s, h)
        except Exception as e_:
            err("k_req", e_)
            r, nC = 0.0, 0
        cap = int(round(math.log2(int(getattr(s, "kcap", 1)))))
        d0 = 0 if r <= 1.0 else min(cap, int(math.ceil(math.log2(math.ceil(r - 1e-9)))))
        if WLOG:
            with open(os.path.join(out, "wc1.jsonl"), "a") as f:
                f.write(json.dumps(dict(t=ST["fi"], kreq=r, nC=nC, d0=d0, cap=cap)) + "\n")
        return d0

    def loop(s, t, h, d0, depth, setK, nfail, ksw):
        """원래 split_step 과 같은 몸통, 시작만 d0 (d0 > 0 시도는 쪼갠 시도라 --ks_rho·--ks_it 덮어쓰기도 같음)."""
        bk = s._bk()
        tries, tot, redo = [], 0.0, False
        base = (s.ga_c, s.nt, s.cg) if ksw else None
        wf = getattr(s, "wf", {}) if ksw else {}
        K = 1
        for dp in range(d0, max(depth, d0) + 1):
            K = 1 << dp
            if dp > d0:
                s._rs(bk)
            if wf.get("ks_ga") is not None:
                s.ga_c = wf["ks_ga"]
            if wf.get("ks_nt"):
                s.nt, s.cg = int(wf["ks_nt"]), int(wf["ks_cg"])
            setK(K)
            redo = s.step(t) or redo
            ms = float(s.last_ms)
            nf = nfail(K)
            tries.append((K, nf, ms))
            tot += ms
            if nf == 0 or not math.isfinite(ms):
                break
        if base is not None:
            s.ga_c, s.nt, s.cg = base
        s.last_ms = tot
        return redo, K, tries

    if NEW:
        from anchorflow import fem_an_rt as ART
        o_split = ART.split_step

        def split_step(s, t, h, depth):
            ST["fi"] = int(round(t / h))
            if s.ga and getattr(s, "ga_first", False):                                    # 원래 함수의 첫 프레임 예외 그대로
                return o_split(s, t, h, depth)
            d0 = depth0(s, h)
            if d0 == 0:
                return o_split(s, t, h, depth)
            return loop(s, t, h, d0, depth, lambda K: ART.set_K_an(s, K, h), lambda K: ART.C3.nfail(s, s.cjl, s.cjg, K), False)
        ART.split_step = split_step
    else:
        from anchorflow import fem_ao_rt as AORT
        o_split = AORT.split_step

        def split_step(s, t, h, sub_dt, depth):
            ST["fi"] = int(round(t / h))
            d0 = depth0(s, h)
            if d0 == 0:
                return o_split(s, t, h, sub_dt, depth)
            return loop(s, t, h, d0, depth, lambda K: AORT.set_K_ga(s, K, h, sub_dt), lambda K: AORT.C3.nfail(s, s.ufl, s.ufg, K), True)
        AORT.split_step = split_step
    print(f"[절 CM59] --wc1k: 접촉 CFL 첫 시도 걸음 수 ({'새' if NEW else '옛'} 실행기, 바닥 법선 {None if NR is None else NR.tolist()}, PN {PN:.6g}, 장벽 두께 d̂ {DH:.6g} m)", flush=True)

sys.argv = [os.path.join(HERE, "ghost6_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")
