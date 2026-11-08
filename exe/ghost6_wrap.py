"""절 CM57 묶음 ghost6 감싸개 (exe/ghost5_wrap.py 복사, 모듈 fem_ghost6): ghost5 인자 + --g6c 0|1 (1 = 보존 판: 강성 벌점 인자 = 기준 형상 변위 u0 + 걸음 변위 y) ·
  --g6en 1 (프레임마다 에너지 진단 -> g6en.jsonl, 결과 불변) · --g6fd 프레임 (u0 커널·아핀 운동·u0 든 힘·에너지·선탐색 검사 -> g6fd.jsonl, 결과 불변).
원래 절 CM55 묶음 ghost5 감싸개 (exe/ghost4_wrap.py 복사, 모듈 fem_ghost5): 인자 --g4* 그대로 + --g5w 방식 (면 가중, 0 = 없음) · --g5wmax w_max (방식 2) · --g5ed 프레임 (헤시안 몫 진단 -> g5ed.jsonl) ;
  --g4dump 덤프에 칸 점유 phi·칸 밀도 rhoc·면 가중 (실행 방식) fw·입자 logJp (있으면)·속도 v 를 더함.
원래 절 CM53 묶음 ghost4 감싸개 (exe/ghost3_wrap.py 복사; 인자 --g4*, 모듈 fem_ghost4, 면 한정 방식 --g4t 0~3, 칸 덤프 --g4dump). 원래: ghost penalty 3 (lib/anchorflow/fem_ghost4.py -- 고스트 질량 + 얇은 구조 면 빼기 + 칸별 커널) 을 붙인 뒤 exe/escape4_wrap.py 로 넘김
(그쪽 --e5r·--e5a, escape3 의 --e3w, escape2·escape1·speed3 인자는 그대로 넘어감).

사용: python exe/ghost3_wrap.py [--g4 g] [--g4m 1] [--g4t 1] [--g4fd 1,5,13] <escape4_wrap 인자...> <실행기.py> <실행기 인자...>
  --g4 g     ghost penalty 3 켬, γ̄ = g (강성 k = g (2μ+λ) h, 질량 k_M = g ρ̄ h³). g = 0 = 커널은 다 돌고 더하는 값 0 (결정적 판 허용, 해시 대조용). 없으면 이 모듈을 안 부름.
  --g4m 0|1  고스트 질량 (기본 1), --g4t 0|1|2|3 면 한정 방식 (기본 1): 0 없음 (절 CM47 면) · 1 3x3x3 열림 두꺼운 칸끼리 (절 CM51) · 2 5x5x5 열림 두꺼운 칸끼리 · 3 내부 칸-경계 칸 면.
  --g4x 0|1  구속 (cuboid 고정) 노드에 닿는 면 빼기 (기본 0).
  --g4dump L (결과 불변) 프레임 목록 L 의 걸음 뒤 칸 덤프 <out>/g4cells_f<프레임>.npz (칸 좌표 cb·경계·두꺼움·면 표지·격자 A·IP·입자 위치·밀도).
  --g4fd L   (결과 불변) 프레임 목록 L 의 걸음 뒤 그래프 밖 검사·비교·시간 (fem_ghost3.check) -> <out>/g3fd.jsonl
  환경 AF_G1_DG=1 (결과 불변): <out>/g1.jsonl 에 프레임별 경계 칸·면 (빼기 뒤·앞)·두꺼운 칸 수.
"""
import atexit
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
CAND = {}
while argv and (argv[0].startswith("--g4") or argv[0].startswith("--g5") or argv[0].startswith("--g6")):
    CAND[argv[0][2:]] = argv[1]
    del argv[:2]
if "g4" in CAND:
    os.environ["AF_G4_GAMMA"] = str(float(CAND["g4"]))
    os.environ["AF_G4_MASS"] = str(int(float(CAND.get("g4m", 1))))
    os.environ["AF_G4_THIN"] = str(int(float(CAND.get("g4t", 1))))
    os.environ["AF_G4_FIXX"] = str(int(float(CAND.get("g4x", 0))))
    os.environ["AF_G5_W"] = str(int(float(CAND.get("g5w", 0))))
    os.environ["AF_G5_WMAX"] = str(float(CAND.get("g5wmax", 3)))
    os.environ["AF_G6_CONS"] = str(int(float(CAND.get("g6c", 0))))
    from anchorflow import fem_ghost6 as G3
    from anchorflow import fem_an_rt as ART
    from anchorflow import fem_an_rt_v0 as V0
    G3.install()
    print(f"[절 CM57] ghost penalty 6 켬 (보존 판 {G3.CONS}: 강성 인자 = {'u0 + y (기준 형상 = 첫 프레임 입자 위치, 입자 F 로 노드까지 아핀 연장)' if G3.CONS else 'y (걸음 변위, ghost5 그대로)'})", flush=True)
    print(f"[절 CM55] ghost penalty 5 켬 (γ̄ {G3.GAMMA:g}, 질량 {G3.MASS}, 면 한정 방식 {G3.THIN}, 구속 노드 면 빼기 {G3.FIXX}, 면 가중 방식 {G3.WMODE} w_max {G3.WMAX:g}, 칸별 커널)", flush=True)
    FED = {int(q) for q in CAND.get("g5ed", "").split(",") if q.strip()}
    FD = {int(q) for q in CAND.get("g4fd", "").split(",") if q.strip()}
    FDUMP = {int(q) for q in CAND.get("g4dump", "").split(",") if q.strip()}
    FD6 = {int(q) for q in CAND.get("g6fd", "").split(",") if q.strip()}
    EN6 = int(float(CAND.get("g6en", 0)))

    def dump(s, fi):
        import numpy as np
        IPl = [int(q) for q in s.IP.cpu().tolist()]
        nc = IPl[11]
        import torch
        torch.cuda.synchronize()
        _ok = lambda s_, nm, n, a, b: s_.k(nm, n, a, b)
        s.g3dummy = torch.zeros(4, device=s.dev, dtype=torch.int32)
        alt = {}
        s._gpin = True
        try:
            for t_ in (0, 1, 2, 3, 4, 5):                                                             # 네 방식 모두의 두꺼움·면 표지 (같은 칸 상태), 끝에 실행 방식으로 되돌림
                fmt = torch.zeros_like(s.g3fm)
                G3.build_mask(s, _ok, t_, fmt, None)
                torch.cuda.synchronize()
                alt[f"thk{t_}"] = s.g3thk[:nc].cpu().numpy().copy(); alt[f"fm{t_}"] = fmt[:nc].cpu().numpy().copy()
                if getattr(s, "fixd", None) is not None:                                        # 구속 노드에 닿는 면 빼기 뒤 표지도 (FIXX 와 무관하게 기록)
                    s.k("gq_fix", int(s.capQc), (G3.P(s.IP), G3.P(s.cb), G3.P(s.fixd), G3.P(fmt), G3.P(None)), 128)
                    torch.cuda.synchronize()
                    alt[f"fmx{t_}"] = fmt[:nc].cpu().numpy().copy()
            G3.build_mask(s, _ok, G3.THIN, s.g3fm, None)
            G3.apply_fix(s, _ok, s.g3fm, None)
            torch.cuda.synchronize()
        finally:
            s._gpin = False
        np.savez_compressed(os.path.join(out, f"g4cells_f{fi}.npz"), IP=np.array(IPl), cb=s.cb.reshape(-1, 3)[:nc].cpu().numpy(), cut=s.g3cut[:nc].cpu().numpy(),
                            thk=s.g3thk[:nc].cpu().numpy(), fm=s.g3fm[:nc].cpu().numpy(), A=s.A.reshape(3, 3).cpu().numpy(), x=s.x.reshape(-1, 3).cpu().numpy(),
                            rho=(s.m / s.vol).cpu().numpy(), thin=G3.THIN, phi=s.g5phi[:nc].cpu().numpy(), rhoc=s.g5rho[:nc].cpu().numpy(),
                            fw=s.g5w[:nc * 3].cpu().numpy(), wmode=G3.WMODE, h=float(s._gph),
                            ljp=(s.ljp.cpu().numpy() if getattr(s, "ljp", None) is not None else np.zeros(0)), v=(s.v.reshape(-1, 3).cpu().numpy() if getattr(s, "v", None) is not None else np.zeros(0)), **alt)
    DG1 = os.environ.get("AF_G1_DG", "0") == "1"
    L1 = open(os.path.join(out, "g1.jsonl"), "w") if DG1 else None
    ST = dict(n=0, done=set())

    def err(tag, e_):
        with open(os.path.join(out, "g4_err.txt"), "a") as f_:
            import traceback
            f_.write(f"{tag}: {e_!r}\n{traceback.format_exc()}\n")

    def torch_ok(s):
        import torch
        return bool(torch.isfinite(s.y).all())

    def wrap_step(cls):
        _os = cls.step

        def step(s, t, *a, **kw):
            d = getattr(s, "_g3dep", 0)
            if d == 0 and G3.CONS and getattr(s, "g6X0", None) is None:                              # 절 CM57: 기준 형상 X0 = 첫 step 앞 입자 위치 (그래프 밖에서 한 번)
                s.g6X0 = s.x.detach().clone().contiguous()
                print(f"[절 CM57] 기준 형상 X0 = 첫 프레임 입자 위치 {tuple(s.g6X0.shape)}", flush=True)
            s._g3dep = d + 1
            try:
                r = _os(s, t, *a, **kw)
            finally:
                s._g3dep = d
            if d != 0 or getattr(s, "gpcnt", None) is None:
                return r
            ST["n"] += 1
            h = float(getattr(s, "h", 0.0) or 0.0)
            fi = int(round(t / h)) + 1 if h > 0 else ST["n"]
            try:
                if L1 is not None:
                    c = s.gpcnt.cpu().tolist()
                    L1.write(json.dumps(dict(f=fi, ncut=int(c[0]), nface=int(c[1]), nface_all=int(c[2]), nthick=int(c[3]), nc=int(s.IP[11].item()),
                                             k=float(s._gpk), kM=float(s._g3kM), it2=float(getattr(s, "_g3it2", 0.0)))) + "\n"); L1.flush()
                if fi in FDUMP:
                    dump(s, fi)
                if EN6:
                    G3.endiag(s, fi, out, 0.1 * float(s._gpkap) * float(s._gph))
                if fi in FD6 and bool(torch_ok(s)):
                    rs6 = G3.check6(s, fi, out)
                    print(f"[절 CM57 검사] 프레임 {fi}: {json.dumps(rs6, ensure_ascii=False)[:1500]}", flush=True)
                if fi in FED:
                    rs5 = G3.ediag(s, fi, out)
                    print(f"[절 CM55 진단] 프레임 {fi}: {json.dumps(rs5, ensure_ascii=False)[:1200]}", flush=True)
                if fi in FD and fi not in ST["done"] and bool(torch_ok(s)):
                    ST["done"].add(fi)
                    rs = G3.check(s, fi, out)
                    print(f"[절 CM53 검사] 프레임 {fi}: {json.dumps(rs, ensure_ascii=False)[:1500]}", flush=True)
            except Exception as e_:
                err(f"프레임 {fi}", e_)
            return r
        cls.step = step

    for cls_ in (ART.RTSolver, V0.RTSolver):
        wrap_step(cls_)

    @atexit.register
    def _info():
        try:
            json.dump(dict(cons=G3.CONS, gamma=G3.GAMMA, mass=G3.MASS, thin=G3.THIN, wmode=G3.WMODE, wmax=G3.WMAX, fixx=G3.FIXX, nk=G3.ST["nk"], said=G3.ST["said"], nmass=G3.ST["nmass"], nstep=G3.ST["nstep"]),
                      open(os.path.join(out, "g6info.json"), "w"), ensure_ascii=False)
        except Exception as e_:
            err("info", e_)

sys.argv = [os.path.join(HERE, "escape4_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")
