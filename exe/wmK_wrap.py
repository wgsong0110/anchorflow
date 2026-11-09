"""절 CM80 묶음 wmK 감싸개 (진단, 기본 판 안 바꿈): 공통 H5 바탕에서 첫 충돌 프레임 하나만 K 걸음 (K = 16·32·64·128) 으로 풂.
절 CM59 N (--wc1k 2: '새로 닿는 입자' C = 프레임 시작에 장벽 띠 밖 d_p > d̂ 이고 프레임 시작 속도로 이 프레임 안에 띠에 들어감, d_p + v_n,p h ≤ d̂, v_n,p < 0)
과 같은 판정으로 C 가 처음 비지 않은 프레임 (= 첫 충돌 프레임) 만 고정 K, 나머지 프레임은 H5 그대로 (K 1 + 기존 쪼개기 깊이 --wmkd, 기본 3 = H5 의 --ksplit 3).

절 CM59 N 과 같은 시도 차례: X3c 끈 채 K 걸음 -> 무너진 걸음 (C3.nfail) 이 있으면 되돌리고 X3c 켠 채 같은 K 걸음 한 번 (N 은 첫 접촉에 K0 = 상한 16 에서 같은 일).
구현: 실행기를 띄우는 순간 (다른 감싸개가 다 설치된 뒤 = 가장 바깥) WRX.split_col 과 AORT.set_K_ga 를 감쌈.
  - 그 프레임만 s._wmK = K 를 켜고 split_col 사슬 (시간 재기 speed6~9 -> speed4 -> ...) 을 깊이 0 으로 부름 -> 사슬 안 set_K_ga(K 1) 이 K 로 바뀜
    (가장 바깥이라 speed7 지연 백업 Z 는 K > 1 로 보고 백업을 실제로 뜸 = 보통 K ≥ 2 쪼갠 시도와 같은 길). 깊이 0 이라 X3c 다시 하기도 K 한 번.
  - 실행기 --ksplit 는 log2 K 이상으로 줘야 함 (버퍼 kcap = 2^ksplit); 다른 프레임의 쪼개기 깊이는 min(실행기 깊이, --wmkd).
  - 예산 T2 는 정의상 K 1 걸음만 자름 (fem_speed3 gate: _tbK == 1) -> 그 프레임은 예산으로 안 잘림 (N 과 같음).
  - fem.jsonl 의 K·ks.tries 는 실제 K 로 바꿔 돌려줌 (사슬 안에서는 K 1 이름표). <out>/wmk.jsonl: 프레임마다 판정 (K_req·C 크기), 그 프레임 시도·GPU ms·벽시계 ms.

사용: python exe/wmK_wrap.py [--wmkk K] [--wmkd 3] [--wmksv 1] <exe/wolfh7_wrap.py 인자 ...> <실행기.py> <실행기 인자...>
  --wmksv 1  진단 (결과 불변, exe/wmcrack1_wrap.py --wc1sv 와 같은 글자): --wsh_sv 프레임마다 입자 탄성 F 특이값·det -> <out>/wsh/fs_<프레임>.npz.
  인자 없으면 아무것도 안 바꿈 (wolfh7_wrap 사슬만).
"""
import json
import math
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
C = dict(k="0", d="3", sv="0")
while argv and argv[0].startswith("--wmk"):
    C[argv[0][5:]] = argv[1]
    del argv[:2]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
KF = int(float(C["k"]))
D0 = int(float(C["d"]))


def _arg(name, default=None):
    """마지막으로 준 값 (argparse 와 같음; H5 낱말의 --ksplit 3 뒤에 판 목록의 --ksplit 이 옴)."""
    return argv[len(argv) - 1 - argv[::-1].index(name) + 1] if name in argv else default


def err(where, e_):
    import traceback
    with open(os.path.join(out, "wmk_err.txt"), "a") as f:
        f.write(f"[{where}] {e_!r}\n{traceback.format_exc()}\n")


if int(float(C["sv"])):                                                                   # exe/wmcrack1_wrap.py --wc1sv 글자 그대로
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
            err(f"wmKsv f{t}", e_)
        return r
    WSHM.Rec.frame = frame
    print("[절 CM80] --wmksv: wsh_sv 프레임에 입자 탄성 F 특이값·det 저장 (결과 불변)", flush=True)

_orig = runpy.run_path
ST = dict(done=False, inst=False)

if KF > 0:
    import torch
    assert KF & (KF - 1) == 0 and KF >= 2, "--wmkk 는 2 의 거듭제곱 (≥ 2)"
    SIM = next((q for q in argv if q.endswith(".py") and "rt_sim" in q), "")
    assert os.path.basename(SIM) == "fem_ao_rt_sim.py", "절 CM80 은 옛 실행기 (수박) 만"
    assert (1 << int(_arg("--ksplit", 0))) >= KF, f"--ksplit 를 log2 {KF} 이상으로 (버퍼 kcap)"
    _d = torch.load(_arg("--data"), map_location="cpu", weights_only=False)
    _cfg = _d["cfg"]
    del _d
    DXG = float(_cfg.get("grid_lim", 2.0)) / int(_cfg.get("n_grid", 100))
    _q = [float(q) for q in _arg("--floor_plane").split(",")]
    NR = torch.tensor(_q[3:], dtype=torch.float64); NR = NR / NR.norm()
    PN = float(NR @ torch.tensor(_q[:3], dtype=torch.float64))
    DH = 0.5 * DXG * float(_arg("--bar_dhmul", 1.0)) if "--fl_bar" in argv else 0.0   # exe/wmcrack1_wrap.py 와 같은 장벽 두께 d̂

    def k_req(s, h):
        """exe/wmcrack1_wrap.py k_req (--wc1k 2) 와 같은 식: 새로 닿는 입자 C 와 K_req = max_C (−v_n) h / Δx."""
        x = s.x.reshape(-1, 3).double(); v = s.v.reshape(-1, 3).double()
        n = NR.to(x.device)
        d = x @ n - PN
        vn = v @ n
        ok = torch.isfinite(d) & torch.isfinite(vn)
        Cm = ok & (vn < 0) & (d + vn * h <= DH) & (d > DH)
        nC = int(Cm.sum())
        if nC == 0:
            return 0.0, 0
        dx = float(s.A.reshape(3, 3)[:, 0].double().norm())
        return float((-vn[Cm]).max()) * h / dx, nC

    def wlog(**kw):
        with open(os.path.join(out, "wmk.jsonl"), "a") as f:
            f.write(json.dumps(kw) + "\n")

    def install():
        from anchorflow import fem_ao_rt as AORT
        from anchorflow import fem_wmrx as WRX
        o_setK = AORT.set_K_ga

        def set_K_ga(s, K, *a, **kw):
            f = int(getattr(s, "_wmK", 0) or 0)
            return o_setK(s, f if f else K, *a, **kw)
        AORT.set_K_ga = set_K_ga
        o_col = WRX.split_col

        def split_col(s, t, h, sub_dt, depth, fn=None):
            fi = int(round(t / h))
            if ST["done"] or fn is not None:
                return o_col(s, t, h, sub_dt, min(int(depth), D0), fn)
            try:
                r, nC = k_req(s, h)
            except Exception as e_:
                err("k_req", e_)
                r, nC = 0.0, 0
            if nC == 0:
                wlog(t=fi, kreq=r, nC=nC, K=None)
                return o_col(s, t, h, sub_dt, min(int(depth), D0), fn)
            ST["done"] = True
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            s._wmK = KF
            try:
                redo, _K, tries = o_col(s, t, h, sub_dt, 0, fn)
            finally:
                s._wmK = 0
            torch.cuda.synchronize()
            wall = 1e3 * (time.perf_counter() - t0)
            tries = [(KF,) + tuple(q[1:]) for q in tries]
            wlog(t=fi, kreq=r, nC=nC, K=KF, sK=int(s.K), tries=[[float(v) if isinstance(v, float) else v for v in q] for q in tries], gpu_ms=float(s.last_ms), wall_ms=wall)
            print(f"[절 CM80 wmK] 첫 충돌 프레임 t {fi} (f{fi + 1}): 새로 닿는 입자 {nC} · K_req {r:.2f} -> K {KF} 고정, 시도 {tries}, GPU {float(s.last_ms):.1f} ms · 벽시계 {wall:.1f} ms", flush=True)
            return redo, KF, tries
        WRX.split_col = split_col
        print(f"[절 CM80] wmK 켬: 첫 충돌 프레임만 K {KF} (다른 프레임 쪼개기 깊이 ≤ {D0}), 바닥 법선 {NR.tolist()} PN {PN:.6g} d̂ {DH:.6g} m", flush=True)

    def run_path(path, *a, **kw):
        if not ST["inst"] and os.path.basename(path) == "fem_ao_rt_sim.py":
            ST["inst"] = True
            install()
        return _orig(path, *a, **kw)
    runpy.run_path = run_path

sys.argv = [os.path.join(HERE, "wolfh7_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")
