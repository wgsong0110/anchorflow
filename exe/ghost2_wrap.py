"""절 CM48 묶음 ghost2 감싸개: 절 CM47 ghost penalty (lib/anchorflow/fem_ghost2.py = fem_ghost1 커널 그대로 + γ 인자 + 검사) 를 붙인 뒤 exe/escape3_wrap.py 로 넘김.

사용: python exe/ghost2_wrap.py [--g2 g] [--g2fd 1,5,13] [--g2sys 1] [--g2fs 1] <escape3_wrap 인자...> <실행기.py> <실행기 인자...>
  --g2 g     ghost penalty 켬, γ_a = (2μ+λ)·g (절 CM47 은 g = 1e-4). g = 0 = 커널은 다 돌고 더하는 값 0 (결정적 판 허용, 해시 대조용). 없으면 이 모듈을 안 부름.
  --g2fd L   (결과 불변) 프레임 목록 L 의 걸음 뒤 그래프 밖 검사 (fem_ghost2.check) -> <out>/g2fd.jsonl
  --g2sys 1  (결과 불변 -- 새 실행기 --lsfast 를 뺌, 절 AN.22 '결과 같음') 프레임마다 뉴턴 반복별 선탐색 기록 (E0, g·δ, 후보 c = 2^-i 의 전체 에너지) -> <out>/g2sys.jsonl
  --g2fs 1   (결과 불변) 프레임마다 경계 칸·면 집합이 앞 프레임과 얼마나 같은지 (절대 칸 좌표 Jaccard) -> <out>/g2fs.jsonl
  환경 AF_G1_DG=1 (결과 불변): <out>/g1.jsonl 에 프레임별 경계 칸 수·면 수 (절 CM47 과 같은 줄).
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
while argv and argv[0].startswith("--g2"):
    CAND[argv[0][2:]] = argv[1]
    del argv[:2]
if "g2" in CAND:
    os.environ["AF_G2_GAMMA"] = str(float(CAND["g2"]))
    from anchorflow import fem_ghost2 as G2
    from anchorflow import fem_an_rt as ART
    from anchorflow import fem_an_rt_v0 as V0
    G2.install()
    print(f"[절 CM48] ghost penalty 켬 (γ_a = (2μ+λ)·{G2.GAMMA_A:g}, k = γ_a h)", flush=True)
    FD = {int(q) for q in CAND.get("g2fd", "").split(",") if q.strip()}
    SYS = int(float(CAND.get("g2sys", 0))) == 1
    FS = int(float(CAND.get("g2fs", 0))) == 1
    DG1 = os.environ.get("AF_G1_DG", "0") == "1"
    if SYS and "--lsfast" in argv:
        argv.remove("--lsfast")
        print("[절 CM48] --g2sys: --lsfast 뺌 (선탐색 후보 모두 계산; 절 AN.22 결과 같음)", flush=True)
    L1 = open(os.path.join(out, "g1.jsonl"), "w") if DG1 else None
    LS = open(os.path.join(out, "g2sys.jsonl"), "w") if SYS else None
    LF = open(os.path.join(out, "g2fs.jsonl"), "w") if FS else None
    ST = dict(n=0, done=set(), prev=None)

    def err(tag, e_):
        with open(os.path.join(out, "g2_err.txt"), "a") as f_:
            import traceback
            f_.write(f"{tag}: {e_!r}\n{traceback.format_exc()}\n")

    def wrap_step(cls):
        _os = cls.step

        def step(s, t, *a, **kw):
            d = getattr(s, "_g2dep", 0)
            s._g2dep = d + 1
            if d == 0 and LS is not None and getattr(s, "DG", None) is not None:
                s.DG.zero_()
            try:
                r = _os(s, t, *a, **kw)
            finally:
                s._g2dep = d
            if d != 0 or getattr(s, "gpcnt", None) is None:
                return r
            ST["n"] += 1
            h = float(getattr(s, "h", 0.0) or 0.0)
            fi = int(round(t / h)) + 1 if h > 0 else ST["n"]
            try:
                if L1 is not None:
                    c = s.gpcnt.cpu().tolist()
                    L1.write(json.dumps(dict(f=fi, ncut=int(c[0]), nface=int(c[1]), nc=int(s.IP[11].item()), k=float(s._gpk))) + "\n"); L1.flush()
                if LS is not None and getattr(s, "DG", None) is not None:
                    D = s.DG.cpu().numpy()
                    nt, K, NC = int(s.nt), int(getattr(s, "last_K", 0) or getattr(s, "Kcur", 1) or 1), int(getattr(s, "ls_n", 10) or 10)
                    rows = []
                    for k_ in range(K):
                        for it in range(nt):
                            r0, r1 = D[2 * (k_ * nt + it)], D[2 * (k_ * nt + it) + 1]
                            rows.append([k_, it, float(r0[1]), float(r1[20] + r1[10]), float(r1[9]), float(r1[14])] + [float(r1[21 + i] + r1[40 + i]) for i in range(NC)])
                    LS.write(json.dumps(dict(f=fi, n=ST["n"], rows=rows)) + "\n"); LS.flush()
                if LF is not None:
                    dd, ST["prev"] = G2.face_change(s, ST["prev"])
                    LF.write(json.dumps(dict(f=fi, n=ST["n"], **dd)) + "\n"); LF.flush()
                if fi in FD and fi not in ST["done"] and bool(torch_ok(s)):
                    ST["done"].add(fi)
                    rs = G2.check(s, fi, out)
                    print(f"[절 CM48 검사] 프레임 {fi}: {json.dumps(rs, ensure_ascii=False)[:1500]}", flush=True)
            except Exception as e_:
                err(f"프레임 {fi}", e_)
            return r
        cls.step = step

    def torch_ok(s):
        import torch
        return bool(torch.isfinite(s.y).all())

    for cls_ in (ART.RTSolver, V0.RTSolver):
        wrap_step(cls_)

    @atexit.register
    def _info():
        try:
            json.dump(dict(gamma=G2.GAMMA_A, nk=G2.ST["nk"], said=G2.ST["said"]), open(os.path.join(out, "g2info.json"), "w"), ensure_ascii=False)
        except Exception as e_:
            err("info", e_)

sys.argv = [os.path.join(HERE, "escape3_wrap.py")] + argv
runpy.run_path(sys.argv[0], run_name="__main__")
