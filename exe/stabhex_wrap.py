"""서브스텝 안정성 '우리(육면체)' 감싸개 (11-12 지시, 묶음 stabhex): 실행기 fem_an_rt_sim.py 의 프레임마다 입자 상태를 읽어 (결과 불변)
안정성 실험 공통 형식 npz (lib/anchorflow/stabstat.py, exe/rep_ip.py --stab_out 과 같은 열) 로 남긴 뒤 인자 첫 감싸개로 넘김.

사용: python exe/stabhex_wrap.py --sh_npz <npz> [--sh_stats 1|0] <다음 감싸개.py 인자...> <실행기.py> <실행기 인자...>
  --sh_stats 0 이면 모듈을 안 부르고 그대로 넘김 (속도 측정 판).
기록: 0 프레임 = --data 의 처음 상태 (x, v, F = I), 프레임 t = 실행기가 프레임마다 한 번 부르는 counts() 순간의 입자 상태
(x = x_orig, v·F 는 perm 역순; 같은 질량 DM.traj_mass·부피 = 질량/밀도·바닥 z = config surface_collider·g).
npz: sub, x [T,32768,3], t (프레임 벽시계 s), keys, stats [T,len(KEYS)], substeps, dt (프레임 h), newton, cg (상한),
it [T-1,6] (프레임별 뉴턴 합·PCG 합·뉴턴 상한 걸음·PCG 상한 풀이·걸음·풀이), frames_done, stop (끝난 까닭).
설치: 실행기를 runpy 로 띄우는 순간 (다른 감싸개가 다 설치된 뒤 = 가장 바깥).
"""
import os
import runpy
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 14000))

argv = sys.argv[1:]
OPT = dict(npz="", stats="1", direct="0")
while argv and argv[0].startswith("--sh_"):
    OPT[argv[0][5:]] = argv[1]
    del argv[:2]
ON = OPT["stats"] not in ("0", "") and OPT["npz"] != ""


def _arg(name, default=None):
    return argv[len(argv) - 1 - argv[::-1].index(name) + 1] if name in argv else default


if "--rr_shape" in argv:                                                              # 루프 안 렌더 (rtrender1) 모형 표에 mic 더함 (rep_ip 와 같은 pgmodel; 결과 불변)
    from anchorflow import rtrender1 as _RR1
    _RR1.MODEL.setdefault("mic", "mic_whitebg-trained")
REC = dict(rows=[], X=[], T=[], IT=[], stop="끝까지", t0=time.time())
from concurrent.futures import ThreadPoolExecutor as _TPE                            # [CM192] 프레임 통계 (탄성 에너지 numpy SVD, 49 만 입자 프레임당 ~1 s CPU -- cProfile 실측) 를 뒤 스레드 여럿으로
_SX = _TPE(max_workers=int(os.environ.get("STABHEX_STAT_W", "6")))
_orig = runpy.run_path
_done = [False]


def _install():
    import numpy as np
    import torch
    from anchorflow import deform_model as DM
    from anchorflow import fem_an_rt as ART
    from anchorflow import stabstat as SS
    d = torch.load(_arg("--data"), map_location="cpu", weights_only=False)
    cfg = d["cfg"]
    m = DM.traj_mass(d, "cpu").double().numpy()
    N = m.shape[0]
    vol = m / float(cfg["density"])
    fl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"]
    FZ = float(fl[0]["point"][2]) if fl else 0.0
    G = [float(q) for q in cfg["g"]]
    EK = dict(vol=vol, E=float(cfg["E"]), nu=float(cfg["nu"]))
    SUB = SS.subset(N)
    x0 = d["x"][0].double().numpy(); v0 = d["v"][0].double().numpy()
    REC["rows"].append(SS.frame_stats(x0, v0, np.tile(np.eye(3), (N, 1, 1)), m, G, FZ, **EK))
    REC["X"].append(x0[SUB].astype(np.float32))
    REC.update(sub=SUB, h=float(cfg["frame_dt"]), N=N)
    del d
    _oc = ART.RTSolver.counts
    _last = [time.time()]

    def counts(s):                                                                    # 실행기가 프레임마다 한 번 (step·x_orig 뒤)
        o = _oc(s)
        xo = s.x_orig()
        vo = torch.empty_like(s.v); vo[s.perm] = s.v
        Fo = torch.empty_like(s.F); Fo[s.perm] = s.F
        xn, vn, Fn = xo.double().cpu().numpy(), vo.double().cpu().numpy(), Fo.reshape(-1, 3, 3).double().cpu().numpy()
        REC["rows"].append(_SX.submit(SS.frame_stats, xn, vn, Fn, m, G, FZ, **EK))     # [CM192] 같은 함수·같은 입력을 뒤 스레드로 (numpy SVD 가 GIL 을 놓아 시뮬과 겹침; 결과 비트 같음)
        REC["X"].append(xn[SUB].astype(np.float32))
        it = o.get("it") or {}
        REC["IT"].append([it.get(k, -1) for k in ("nN", "nP", "capN", "capP", "steps", "solves")])
        t = time.time(); REC["T"].append(t - _last[0]); _last[0] = t
        return o
    ART.RTSolver.counts = counts
    if OPT["direct"] not in ("0", ""):
        _direct()
    print(f"[stabhex] 안정성 기록 설치: 입자 {N}, 부분표본 {len(SUB)}, 바닥 z {FZ}, -> {OPT['npz']}", flush=True)


def _direct():
    """--sh_direct 1: 프레임 CUDA 그래프 대신 같은 record() 를 매 프레임 바로 띄움 (큰 N 에서 걸음 K 개를 펼친 그래프 캡처가 GPU 메모리 부족).
    나눈 그래프 (--xg, fem_common3) 의 건너뛰기는 커널이 CT 를 보고 스스로 돌아가는 것이라 (그 모듈 설명) 결과 정의는 같음; 넘침 다시 하기는 step() 그대로."""
    from anchorflow import fem_common3 as C3

    class _D:
        def __init__(self, s):
            self.s = s

        def replay(self):
            self.s.record(0.0, self.s.Kcur)

    def capture(s):
        g = _D(s)
        s.graph = g
        s.graphs[s.Kcur] = g
    C3.capture = capture
    print("[stabhex] 그래프 없이 바로 띄움 (--sh_direct)", flush=True)


def _save():
    if not ON or not REC["X"]:
        return
    import numpy as np
    from anchorflow import stabstat as SS
    K = SS.KEYS
    os.makedirs(os.path.dirname(os.path.abspath(OPT["npz"])), exist_ok=True)
    tmp = OPT["npz"] + ".tmp.npz"
    np.savez_compressed(tmp, sub=REC["sub"], x=np.stack(REC["X"]).astype(np.float32), t=np.array(REC["T"]), keys=np.array(K),
                        stats=np.array([[r.get(k, np.nan) for k in K] for r in [q.result() if hasattr(q, "result") else q for q in REC["rows"]]]), substeps=int(_arg("--K", 1)), dt=REC["h"],
                        newton=int(_arg("--newton", -1)), cg=int(_arg("--cg", -1)), tolN=float(_arg("--tolN", -1)), tolP=float(_arg("--tolP", -1)),
                        it=np.array(REC["IT"], dtype=np.float64).reshape(-1, 6), frames_done=len(REC["X"]) - 1, stop=REC["stop"])
    os.replace(tmp, OPT["npz"])
    print(f"[stabhex] 저장 {OPT['npz']}: 프레임 {len(REC['X']) - 1}, 끝 '{REC['stop']}'", flush=True)


def run_path(path, *a, **kw):
    if ON and not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
        _done[0] = True
        _install()
    return _orig(path, *a, **kw)


def _term(signum, frame):
    REC["stop"] = f"신호 {signum}"
    raise SystemExit(128 + signum)


if ON:
    runpy.run_path = run_path
    signal.signal(signal.SIGTERM, _term)
sys.argv = [os.path.abspath(argv[0])] + argv[1:]
try:
    _orig(sys.argv[0], run_name="__main__")
except BaseException as ex:                                                          # 예외로 죽어도 그때까지 기록은 남김 (발산 판정은 분석에서)
    if REC["stop"] == "끝까지":
        REC["stop"] = f"예외 {type(ex).__name__}: {str(ex)[:200]}"
    _save()
    raise
_save()
