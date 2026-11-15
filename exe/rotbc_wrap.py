"""PhysGaussian `enforce_particle_velocity_rotation` (회전 구동기) 와 `enforce_particle_translation` (탄성 판) 을 우리 판 노드 속도 구속으로 거는 감싸개 (paper_remeasure2).

사용: python -u exe/rotbc_wrap.py --rot_bc 1 [<다음 감싸개.py> <그 인자...>] exe/fem_an_rt_sim.py <실행기 인자 ...>
--rot_bc 가 없거나 0 이면 아무것도 안 함. 이 두 경계조건이 없는 씬 (ficus·wolf·bread·레고) 은 켜도 결과 불변 (하는 일이 없음).

정의 (PG mpm_solver_warp.enforce_particle_velocity_rotation / mpm_utils.selection_enforce_particle_velocity_cylinder 그대로):
  영역 = 법선 n 의 원기둥 (|(x−p)·n| < 반 높이 , 법선 직교 거리 < 반지름); 입자 선택은 처음 한 번 (처음 위치).
  시각 start <= t < end 에 영역 입자 속도 = rotation_scale · ((x−p)_⊥ × n) + translation_scale · n  (x 는 현재 위치).
우리 구현 (cut BC 를 노드 상자 구속으로 옮긴 exe/cutbc_wrap.py 와 같은 방식, 새 CUDA 커널 하나):
  - 영역 = 격자 노드 기준 위치가 원기둥 안인 노드 (rt_bc_box 가 상자 노드를 고르는 것과 같은 기준) -> fixd = 1, vbc = 속도 (Dirichlet; 걸음 변위 증분 τ·vbc).
    속도의 반지름 벡터는 노드의 현재 위치 (기준 위치 + 변위 U) 로 -- PG 가 현재 입자 위치를 쓰는 것과 같음.
  - 한 프레임 안에서 구간이 바뀌면 (vasedeck: 0.3·0.6·0.9 s 는 프레임 0.04 s 의 정수배가 아님) 프레임 시각 [t, t+h) 와 겹친 몫 f 만큼 속도를 곱해 더함
    (같은 영역의 구간 여러 개는 합쳐 한 구속으로: 속도 = Σ 구간 속도 · f, 어느 구간이든 겹치면 영역은 고정).
  - 탄성 판 (재질 판이 아님) 에서는 enforce_particle_translation 도 노드 속도 구속으로 -- 기존 --cfg_bc 는 cuboid 만 읽어 이를 놓쳤다.
    재질 판 (--pbc) 의 translation 은 기존 TREL/PVB 경로가 처리하므로 건드리지 않음.
    정의 (11-15 고침, PG mpm_solver_warp.enforce_particle_velocity_translation / mpm_utils.selection_enforce_particle_velocity_translation 그대로):
      입자 선택은 처음 한 번 (처음 입자 위치가 상자 안인 입자), 시각 start <= t < end 에 그 입자들의 속도 = velocity (P2G 앞).
      상자는 입자를 고르는 데만 쓴다 -- 공간 영역이 아니다 (나중에 그 상자로 들어온 다른 입자는 구속되지 않는다).
    우리 구현 = 입자 구속을 노드로 옮기는 공통 규칙 (lib/anchorflow/fem_rt.cfg_bc_nodes = train_deform --cfg_bc 와 같음):
      구속 입자의 질량 가중 B-스플라인 몫 (프레임 시작 위치) 이 노드 질량의 절반을 넘는 노드 -> fixd = 1, vbc = 구속 입자 질량 가중 평균 속도.
    (11-15 이전 판은 이 상자를 격자 노드 기준 위치로 골라 (rt_bc_box) 상자 공간 전체를 얼렸다 -- pillow2sofa 에서 상자 z < 0.9 로
     떨어져 들어가야 할 방석이 상자 윗면 (바구니 테두리 높이) 에 걸렸다.)
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 15000))

KERNEL = r'''
// [rotbc_wrap] PG enforce_particle_velocity_rotation: 원기둥 안 노드 (기준 위치) -> fixd, vbc = rs (r_⊥ x n) + ts n (r 은 현재 위치 = 기준 + U)
// RP 한 구속 12 칸: 점 3, 단위 법선 3, 반 높이, 반지름, 회전 몫 rs, 이동 몫 ts, (빈 칸), 켜짐
__global__ void rt_bc_rot(const int* IP, const float* Am, const float* U, const float* RP, int ib, unsigned char* fixd, float* vbc) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    const float* b = RP + ib*12;
    if (b[11] == 0.f) return;
    int nb1 = IP[7], nb2 = IP[8];
    int i = n / (nb1*nb2), j = (n / nb2) % nb1, k = n % nb2;
    float yq[3] = {(float)(IP[0] + i), (float)(IP[1] + j), (float)(IP[2] + k)};
    float xr[3]; for (int a=0;a<3;a++) xr[a] = Am[a*3]*yq[0] + Am[a*3+1]*yq[1] + Am[a*3+2]*yq[2];
    float o[3]; for (int a=0;a<3;a++) o[a] = xr[a] - b[a];
    float ax = o[0]*b[3] + o[1]*b[4] + o[2]*b[5];
    float r0 = o[0] - ax*b[3], r1 = o[1] - ax*b[4], r2 = o[2] - ax*b[5];
    if (!(fabsf(ax) < b[6] && sqrtf(r0*r0 + r1*r1 + r2*r2) < b[7])) return;
    float c[3]; for (int a=0;a<3;a++) c[a] = xr[a] + U[n*3+a] - b[a];
    float ac = c[0]*b[3] + c[1]*b[4] + c[2]*b[5];
    float q0 = c[0] - ac*b[3], q1 = c[1] - ac*b[4], q2 = c[2] - ac*b[5];
    float v0 = b[8]*(q1*b[5] - q2*b[4]) + b[9]*b[3];
    float v1 = b[8]*(q2*b[3] - q0*b[5]) + b[9]*b[4];
    float v2 = b[8]*(q0*b[4] - q1*b[3]) + b[9]*b[5];
    fixd[n] = 1; vbc[n*3] = v0; vbc[n*3+1] = v1; vbc[n*3+2] = v2;
}
// [rotbc_wrap 11-15] PG enforce_particle_translation (입자 선택 구속) -> 노드: 구속 입자 질량 몫 > 노드 질량 절반 -> fixd, vbc = 구속 입자 질량 가중 속도
// TP 한 구속 4 칸: 속도 3, 켜짐. pid[원래 입자 번호] = 구속 번호 (-1 = 없음, 겹치면 config 뒤쪽). perm[p] = 정렬 입자 p 의 원래 번호.
// PA 노드 5 칸: 전체 질량, 구속 질량, 구속 질량 x 속도 3. 프레임 시작 위치 x 에서 이차 B-스플라인 (rt_p2g 와 같은 base·가중).
__global__ void rt_trp_zero(const int* IP, float* PA) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    for (int a=0;a<5;a++) PA[n*5+a] = 0.f;
}
__global__ void rt_trp_acc(int N, const int* IP, const float* Ai, const float* x, const float* m, const long long* perm, const int* pid, const float* TP, float* PA) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int ib = pid[perm[p]];
    bool on = ib >= 0 && TP[ib*4+3] != 0.f;
    int bs[3]; float wx[3][3], dw[3][3];
    for (int a=0;a<3;a++) {
        float y = Ai[a*3]*x[p*3] + Ai[a*3+1]*x[p*3+1] + Ai[a*3+2]*x[p*3+2];
        float fb = floorf(y - 0.5f);
        bs[a] = (int)fb - IP[a];
        bwf(y - fb, wx[a], dw[a]);
    }
    int nb1 = IP[7], nb2 = IP[8];
    float mp = m[p];
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((bs[0]+i)*nb1 + (bs[1]+j))*nb2 + (bs[2]+k);
        float ww = wx[0][i]*wx[1][j]*wx[2][k]*mp;
        atomicAdd(&PA[nid*5], ww);
        if (on) { atomicAdd(&PA[nid*5+1], ww); for (int a=0;a<3;a++) atomicAdd(&PA[nid*5+2+a], ww*TP[ib*4+a]); }
    }
}
__global__ void rt_trp_fix(const int* IP, const float* PA, unsigned char* fixd, float* vbc) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float mt = PA[n*5], mc = PA[n*5+1];
    if (mc > 0.f && mc > 0.5f*mt) { fixd[n] = 1; for (int a=0;a<3;a++) vbc[n*3+a] = PA[n*5+2+a]/mc; }
}
'''


def install():
    import numpy as np
    import torch
    from anchorflow import fem_an_rt as ART
    anchor = "// 선탐색 (융합): 후보 c = ¼"
    assert anchor in ART.SRC, "[rotbc] 커널 삽입 위치를 못 찾음"
    ART.SRC = ART.SRC.replace(anchor, KERNEL + anchor, 1)
    orig_init, orig_k, orig_set, orig_alloc = ART.RTSolver.__init__, ART.RTSolver.k, ART.RTSolver.set_bc, ART.RTSolver.alloc

    def init(s, *a, **kw):
        g = None
        f = sys._getframe(1)
        while f is not None:
            if "cfg" in f.f_globals and "h" in f.f_globals and f.f_globals.get("__name__") == "__main__":
                g = f.f_globals; break
            f = f.f_back
        assert g is not None, "[rotbc] 실행기 전역 (cfg) 을 못 찾음"
        cfg = g["cfg"]
        bcs = cfg.get("boundary_conditions") or []
        rot = [b for b in bcs if b.get("type") == "enforce_particle_velocity_rotation"]
        mat_path = kw.get("mat") is not None or kw.get("trel") is not None or kw.get("pvb") is not None
        tr = [] if mat_path else [b for b in bcs if b.get("type") == "enforce_particle_translation"]
        lo = g.get("LOFF")
        assert not ((rot or tr) and lo is not None and float(lo.abs().max()) > 0), "[rotbc] 띄움 (--lift) 과 같이는 안 됨"
        bc = list(kw.get("bc") or [])
        s._trb = [([float(q) for q in b["velocity"]], float(b.get("start_time", 0.0)), float(b.get("end_time", 1e3))) for b in tr]
        groups = {}
        for b in rot:
            nv = np.asarray(b["normal"], dtype=np.float64); nv = nv / np.linalg.norm(nv)
            hh, rr = (float(q) for q in b["half_height_and_radius"])
            key = (tuple(float(q) for q in b["point"]), tuple(nv.tolist()), hh, rr)
            groups.setdefault(key, []).append((float(b["rotation_scale"]), float(b["translation_scale"]), float(b.get("start_time", 0.0)), float(b.get("end_time", 1e3))))
        if (groups or tr) and not bc:
            bc.append(([1e9] * 3, [1e-9] * 3, [0.0] * 3, 1e9, 2e9))            # 켜지지 않는 더미 상자: fixd·vbc 버퍼와 rt_bc_box 호출 자리를 만들기 위함
        kw["bc"] = bc or None
        s._rbc_allow = bool(groups)                                              # 재질 판 (fem_mat_rt.MatRT) 의 "고정 상자 없는 씬만" 검사 예외
        if tr:                                                                     # 입자 선택: 처음 위치 (원래 입자 순서) 가 상자 안 -> 구속 번호 (뒤 구속이 덮어씀, PG 처럼 뒤 수정자가 이김)
            x0 = (a[0] if a else kw["x"]).to("cuda").double()
            pid = torch.full((x0.shape[0],), -1, device="cuda", dtype=torch.int32)
            for ib, b in enumerate(tr):
                msk = ((x0 - torch.tensor(b["point"], device="cuda", dtype=torch.float64)).abs() < torch.tensor(b["size"], device="cuda", dtype=torch.float64)).all(-1)
                pid[msk] = ib
            s.trpid = pid.contiguous()
            s.TP = torch.zeros(len(tr) * 4, device="cuda", dtype=torch.float32)
            print(f"[rotbc] translation 입자 선택 (PG 정의): 구속 {[int((pid == ib).sum()) for ib in range(len(tr))]} / {x0.shape[0]} 입자 -> 노드 (구속 질량 > 절반)", flush=True)
            del x0
        orig_init(s, *a, **kw)
        s._rbc = list(groups.items())
        s.RP = torch.zeros(max(1, len(s._rbc)) * 12, device="cuda", dtype=torch.float32)
        print(f"[rotbc] 회전 구동기 {len(rot)} 개 (영역 {len(s._rbc)} 곳으로 합침), 탄성 판 translation 상자 {len(tr)} 개", flush=True)

    def k(s, name, n, args, b=128):
        if name == "rt_bc_box" and getattr(s, "_trb", None) and int(args[3]) == 0:   # fixd 비운 뒤 첫 상자 앞 (PG: 입자 수정자 -> P2G -> 격자 cuboid 순서라 cuboid 가 이김)
            assert s.PA.numel() >= s.capN * 5, "[rotbc] PA 용량"
            orig_k(s, "rt_trp_zero", n, (args[0], ART.P(s.PA)), b=256)
            orig_k(s, "rt_trp_acc", s.N, (np.int32(s.N), args[0], ART.P(s.Ai), ART.P(s.x), ART.P(s.m), ART.P(s.perm), ART.P(s.trpid), ART.P(s.TP), ART.P(s.PA)), b=128)
            orig_k(s, "rt_trp_fix", n, (args[0], ART.P(s.PA), args[4], args[5]), b=256)
        if name == "rt_bc_box" and getattr(s, "_rbc", None) and int(args[3]) == len(s.bc) - 1:
            for jj in range(len(s._rbc)):
                orig_k(s, "rt_bc_rot", n, (args[0], args[1], ART.P(s.U), ART.P(s.RP), np.int32(jj), args[4], args[5]), b=256)
        return orig_k(s, name, n, args, b)

    def set_bc(s, t):
        orig_set(s, t)
        if getattr(s, "_trb", None):
            v = []
            for (vel, t0, t1) in s._trb:
                v += list(vel) + [1.0 if (t0 <= t < t1) else 0.0]
            s.TP.copy_(torch.tensor(v, dtype=torch.float32))
        if getattr(s, "_rbc", None):
            v = []
            for (pt, nv, hh, rr), wins in s._rbc:
                rs = ts = fs = 0.0
                for (r_, t_, t0, t1) in wins:
                    f = max(0.0, min(t + s.h, t1) - max(t, t0)) / s.h
                    rs += r_ * f; ts += t_ * f; fs += f
                v += list(pt) + list(nv) + [hh, rr, rs, ts, 0.0, 1.0 if fs > 0 else 0.0]
            s.RP.copy_(torch.tensor(v, dtype=torch.float32))

    def alloc(s, need):
        orig_alloc(s, need)
        if getattr(s, "_trb", None):                                               # 노드 용량이 바뀌면 (넘침 다시 할당) 같이 -- 그래프 밖
            s.PA = torch.zeros(s.capN * 5, device="cuda", dtype=torch.float32)

    ART.RTSolver.__init__, ART.RTSolver.k, ART.RTSolver.set_bc, ART.RTSolver.alloc = init, k, set_bc, alloc


if __name__ == "__main__":
    argv = sys.argv[1:]
    ON = 0
    while argv and argv[0] == "--rot_bc":
        ON = int(float(argv[1]))
        del argv[:2]
    _orig = runpy.run_path
    _done = [False]

    def run_path(path, *a, **kw):
        if ON and not _done[0] and os.path.basename(path) == "fem_an_rt_sim.py":
            _done[0] = True
            install()
        return _orig(path, *a, **kw)

    runpy.run_path = run_path
    sys.argv = [os.path.abspath(argv[0])] + argv[1:]
    _orig(sys.argv[0], run_name="__main__")
