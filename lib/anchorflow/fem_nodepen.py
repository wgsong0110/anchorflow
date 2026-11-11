"""절 CM111 진단 (읽기만, 결과 불변): 프레임마다 활성 노드 (질량 > 0) 의 상자 번호·질량·걸음 끝 U 와 입자의 받침 base·분수 좌표를 <out>/nodes/f<프레임>.npz 에.
exe/nodepen_wrap.py 가 설치. 풀이 상태를 안 바꾸고 (읽기, 그래프에 U 사본 복사 몇 개만 더함) 결과 비트가 같다.
 - 프레임 f 의 파일 = 프레임 f-1 -> f 걸음 (걸음 쪼개기로 여러 번 풀면 마지막 = 받아들인 시도가 덮어씀).
 - 노드 번호 = ((i * nb1) + j) * nb2 + k (상자 안), 상자 원점 ip[0:3], 크기 ip[6:9]; 노드 걸음 시작 위치 = A (번호 + 원점), 걸음 끝 위치 = + U.
 - Ust[k] = k 번째 걸음 끝 U (K 걸음; 마지막 = 프레임 끝 U). 걸음 중간 = 그래프 안 다음 걸음 첫 커널 (rt_ga_step0) 직전에 U 를 DNU 에 복사.
 - 입자 pb (int16, 상자 원점 기준)·fx (float16) 는 원래 입자 순서 (pred.pt 와 같은 순서).
"""
import os

import numpy as np
import torch

CFG = {}


def install(out):
    from . import fem_an_rt as ART
    CFG["out"] = out
    os.makedirs(os.path.join(out, "nodes"), exist_ok=True)
    R = ART.RTSolver
    oalloc, ok, orec, ostep = R.alloc, R.k, R.record, R.step

    def alloc(s, *a, **kw):
        r = oalloc(s, *a, **kw)
        if hasattr(s, "DNU"):
            del s.DNU
        torch.cuda.empty_cache()
        s.DNU = torch.zeros(max(s.K, s.Kc or 0, getattr(s, "kcap", 0)), s.capN, 3, device=s.U.device, dtype=torch.float32)
        s._np_st = 0
        return r

    def record(s, *a, **kw):
        s._np_st = 0
        return orec(s, *a, **kw)

    def k(s, name, *a, **kw):
        if name == "rt_ga_step0" and hasattr(s, "DNU"):
            j = s._np_st
            if j >= 1 and j - 1 < s.DNU.shape[0]:
                s.dcopy(s.DNU[j - 1], s.U)                                             # 앞 걸음 끝 U (다음 걸음 첫 커널 직전)
            s._np_st = j + 1
        return ok(s, name, *a, **kw)

    def step(s, t):
        redo = ostep(s, t)
        try:
            if bool(torch.isfinite(s.x[:1]).all()):
                dump(s, t)
        except Exception as ex:                                                        # 진단 실패는 풀이에 영향 없음
            with open(os.path.join(CFG["out"], "nodepen_err.txt"), "a") as f:
                f.write(f"t {t}: {ex!r}\n")
        return redo

    R.alloc, R.k, R.record, R.step = alloc, k, record, step


def dump(s, t):
    fr = int(round(t / s.h)) + 1
    nn = int(s.IP[9]); K = int(s.last_K)
    act = torch.nonzero(s.mn[:nn] > 0).squeeze(1)
    pb = torch.empty_like(s.Np); pb[s.perm] = s.Np
    fx = torch.empty_like(s.fx); fx[s.perm] = s.fx
    Ust = torch.cat([s.DNU[:K - 1, :nn][:, act], s.U[:nn][act][None]], 0) if K > 1 else s.U[:nn][act][None]
    np.savez(os.path.join(CFG["out"], "nodes", f"f{fr:03d}.npz"), ip=s.IP[:14].cpu().numpy(), A=s.A.float().reshape(3, 3).cpu().numpy(), act=act.int().cpu().numpy(),
             mn=s.mn[:nn][act].cpu().numpy(), Ust=Ust.cpu().numpy(), pb=pb.short().cpu().numpy(), fx=fx.half().cpu().numpy(), redo=0, ksK=K)
