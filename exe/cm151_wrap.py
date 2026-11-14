"""절 CM151 감싸개 (진단, 결과 불변): exe/finterp_wrap.py 와 같은 규약 (--fi N --fidg 1 먼저) 에 더해, 프레임 10·20·30 끝에서 가우스 점별 상태를 out/cm151_f<프레임>.npz 로.
점별: xq0 (프레임 시작 위치), mq, Vq, J0 (걸음 시작 P2G F_q 의 det), JX (노드 변위 A F_q 의 det = 이 걸음 노드 변위장이 칸을 늘린/줄인 몫, 반환 사상 앞), JE (반환 사상 뒤),
pc (그 점이 든 칸의 입자 수), cb (칸 정수 좌표).  --fidg 1 이 있어야 J0·JX 가 채워짐 (절 CM92). 읽기만."""
import os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
O = dict(fi="0", fidg="0")
while argv and argv[0] in ("--fi", "--fidg"):
    O[argv[0][2:]] = argv[1]; del argv[:2]
MODE, DG = int(float(O["fi"])), int(float(O["fidg"]))
assert DG, "cm151_wrap 은 --fidg 1 필요"
_e = os.environ.get("CM151_FR", "10,20,30")
FR = tuple(range(int(_e.split("-")[0]), int(_e.split("-")[1]) + 1)) if "-" in _e else tuple(int(v) for v in _e.split(","))
_orig = runpy.run_path; _done = [False]
def run_path(path, *a, **kw):
    if not _done[0] and os.path.basename(path) in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _done[0] = True
        from anchorflow import fem_finterp as FI
        from anchorflow import fem_an_rt as ART
        import numpy as np, torch
        FI.install(out, MODE, DG)
        ostep = ART.RTSolver.step
        def step(s, *aa, **kk):
            r = ostep(s, *aa, **kk)
            try:
                fr = FI.ST["nframe"]
                if fr in FR and getattr(s, "fiJX", None) is not None:
                    nq = int(s.IP[12].item()); N = s.N
                    JE = torch.linalg.det(s.Fe[:nq, 18:27].reshape(nq, 3, 3).double()).float()
                    c = s.cidx[s.pcd[:N].long()].long(); c = c[c >= 0]
                    cnt = torch.bincount(c, minlength=nq // 8 + 1)
                    pc = cnt[(torch.arange(nq, device=cnt.device) >> 3).clamp(max=cnt.numel() - 1)]
                    cn = lambda t: t.detach().cpu().numpy()
                    np.savez(os.path.join(out, f"cm151_f{fr}.npz"), xq=cn(s.xq0[:nq]), mq=cn(s.mq[:nq]), Vq=cn(s.Vq[:nq]), J0=cn(s.fiJ0[:nq]), JX=cn(s.fiJX[:nq]), JE=cn(JE), pc=cn(pc), cb=cn(s.cb[: nq // 8 + 1]),
                             px=cn(s.x[:N]) if hasattr(s, "x") else np.zeros(0),
                             U=cn(s.U), mn=cn(s.mn), fixd=cn(s.fixd) if s.fixd is not None else np.zeros(0), pm=cn(s.pm), nbs=np.array([int(s.IP[7].item()), int(s.IP[8].item())]))   # 절 CM153: 노드 변위·질량·구속
            except Exception as e:
                print("[cm151] 기록 실패", repr(e), flush=True)
            return r
        ART.RTSolver.step = step
    return _orig(path, *a, **kw)
runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "wolfh11_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")
