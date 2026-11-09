"""절 CM90 수렴 판 덤프 (묶음 wmconv; 결과 불변 기록): 가벼운 판 ('--s6l 1', exe/s6 실행기 사본) 은 프레임 고리가 쪼개기 함수만 부르고
`continue` 해서 실행기 --dump 의 F 와 --dump_lj 의 logJp 칸이 채워지지 않음 (빈 메모리 그대로 저장됨). 그래서 실행기 덤프 플래그는 안 쓰고
fem_speed6.Defer 를 감싸 프레임 끝 입자 F (원래 순서) 를 Defer 와 같은 방식 (시뮬 스트림 장치 사본 -> 복사 스트림에서 순서 되돌리기·고정 메모리)
으로 쌓고, finish 뒤 Defer 의 위치 X·logJp LJ 와 함께
  <out>/pred.pt = {"pred": [n, N, 3], "F": [n, N, 3, 3]}   (실행기 --dump 와 같은 꼴)
  <out>/lj.pt   = {"logJp": [n, N]}                         (실행기 --dump_lj 와 같은 꼴)
을 씀 (n = 끝에서 판정한 프레임 수). 풀이 상태는 읽기만.
"""
import os

import torch

_ST = {}


def install(out):
    from . import fem_speed6 as S6
    D0 = S6.Defer
    ok_init, ok_keep, ok_fin = D0.__init__, D0.keep, D0.finish

    def init(D, S, L, *a, **kw):
        ok_init(D, S, L, *a, **kw)
        Np = int(S.x.shape[0])
        D.wcF = torch.empty(L, Np, 9, dtype=S.F.dtype, pin_memory=True)
        D.wcR = [torch.empty((Np, 9), dtype=S.F.dtype, device=S.F.device) for _ in range(3)]
        D.wcE = [None] * 3
        assert D.LJ is not None, "절 CM90 wmconv: Defer 가 logJp 를 안 쌓음 (수박 재질 7 판만)"
        print(f"[절 CM90 wmconv] 덤프 켬: F 고정 메모리 {D.wcF.numel() * D.wcF.element_size() / 1e9:.2f} GB (위치·logJp 는 Defer 것)", flush=True)

    def keep(D, fi, S, redo, K, tries):
        r = ok_keep(D, fi, S, redo, K, tries)
        assert not bool(getattr(S, "det", False)), "절 CM90 wmconv: 비결정 판만 (perm 고정)"
        cur = torch.cuda.current_stream()
        k = fi % 3
        if D.wcE[k] is not None:
            cur.wait_event(D.wcE[k])
        D.wcR[k].copy_(S.F.reshape(S.F.shape[0], 9))
        ev = torch.cuda.Event()
        ev.record(cur)
        D.cs.wait_event(ev)
        with torch.cuda.stream(D.cs):
            P = D.PR[D.p0]
            fo = torch.empty_like(D.wcR[k])
            fo[P] = D.wcR[k]
            D.wcF[fi].copy_(fo, non_blocking=True)
            e2 = torch.cuda.Event()
            e2.record(D.cs)
        D.wcE[k] = e2
        return r

    def finish(D, *a, **kw):
        r = ok_fin(D, *a, **kw)
        try:
            torch.cuda.synchronize()
            D.cs.synchronize()
            n = len(r[0])
            Np = D.Np
            torch.save({"pred": D.X[:n], "F": D.wcF[:n].reshape(n, Np, 3, 3)}, os.path.join(out, "pred.pt"))
            torch.save({"logJp": D.LJ[:n]}, os.path.join(out, "lj.pt"))
            fin = torch.isfinite(D.wcF[:n]).all(-1).all(-1)
            print(f"[절 CM90 wmconv] 덤프 {n} 프레임 -> pred.pt (위치·F) · lj.pt (logJp); F 유한 프레임 {int(fin.sum())}/{n}", flush=True)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            with open(os.path.join(out, "wc_err.txt"), "a") as f:
                f.write(f"finish {ex!r}\n")
        return r

    D0.__init__, D0.keep, D0.finish = init, keep, finish
    print("[절 CM90 wmconv] 가벼운 판 Defer 에 F 덤프 감쌈 (실행기 --dump·--dump_lj 대신)", flush=True)
