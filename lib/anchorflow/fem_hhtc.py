"""절 CM58 묶음 hhtc (조정자 11-08 위임): 수박 첫 접촉 뒤 몸 전체 튀어오름 (절 CM57.4) 을 접촉 걸음의 HHT 걸음 끝 갱신에서 끊는 정의.

원인 (절 CM57.4, 확인됨): 첫 접촉 프레임의 K 1 걸음이 (원자 합 순서 잡음으로) 수렴 판정되면 쪼개지 않고 받는다. 그 풀이는 몸을 바닥에서 한 걸음 안에 멈춰 세운
y 이고, HHT (Newmark γ 0.676, β 0.34) 의 걸음 끝 속도 v1 = α (y − z0) 는 '자리는 멈췄는데 걸음 평균 속도 ≈ 0' 인 걸음에서 들어온 속도를 거의 그대로 뒤집는다
(사다리꼴 규칙의 성질: v1 ≈ −(γ/β − 1) v0; 실측 −7.35 -> +5.39 m/s). 그 걸음의 가속도 a1 = (v1 − v0 − dt (1−γ) a0)/(γ dt) 가 기억으로 넘어가
접촉이 끝난 다음 걸음에 (1−γ) dt a1 = +6 m/s 를 한 번 더 더한다 (에너지 5.0 -> 18.5 J). 보통 판은 첫 시도가 무너져 쪼개기 (K 2) 로 넘어가 이 일이 없다.

정의 (모두 위상·기하 판정, 문턱·클램프 없음; --hc M, 0 이면 진단만 = 결과 불변):
  접촉 노드 판정 (걸음 끝 갱신 바로 앞, 그래프 안): 장벽 표본점 q (활성 AF_BACT, 질량 > 0) 의 부호 거리 d 를 걸음 시작 위치 (U) 와 걸음 끝 위치 (U + thinv (y − U),
  풀이 장벽 rt_bar_q 와 같은 식·같은 기저 -- W̄ (--e3w) 판이면 W̄, R2·A2 의 옮긴 기준 AF_BX 그대로) 에서 재고, d < d̂ (장벽 받침 안 = 장벽 힘이 걸리는 자리)
  인 표본점의 기저가 0 아닌 노드를 '걸음 시작 접촉'·'걸음 끝 접촉' 노드로 표시. 새 접촉 노드 = 걸음 끝 접촉 ∧ ¬ 걸음 시작 접촉 (이 걸음에 접촉 활성 집합에 든 노드).
  충돌 걸음 = 새 접촉 노드가 하나라도 있는 걸음.
  1 (H1, 충돌 걸음 몸 전체 후퇴 오일러): 충돌 걸음은 모든 노드의 걸음 끝 갱신을 후퇴 오일러로 (U1 = y, v1 = (y − U)/h, 가속도 기억 0 -- 절 CM24 --ue be 와 같은 식).
     충돌 아닌 걸음은 그대로 (HHT, 덜 풀린 걸음은 --ue_loc 집합 S 후퇴 오일러 그대로).
  2 (H2, 새 접촉 노드만 후퇴 오일러): 새 접촉 노드만 위 후퇴 오일러, 나머지 그대로.
  3 (H3, 걸음 끝 접촉 노드 가속도 기억 0): 걸음 끝 접촉 노드 (장벽 힘이 걸린 노드) 는 위치·속도 HHT 그대로, 가속도 기억만 0 (절 CM24 --ue a0 와 같은 식).
  4 (H4, H1 을 '적분이 만든 되튐' 걸음으로 좁힘): 충돌 걸음 중 HHT 걸음 끝 속도 v1 = α (y − z0) 가 새 접촉 노드들의 법선 속도 합을 걸음 시작의 들어가는 쪽
     (Σ v0·n < 0) 에서 나오는 쪽 (Σ v1·n > 0) 으로 뒤집는 걸음만 몸 전체 후퇴 오일러 (H1 은 쉬거나 미끄러지며 받침에 새로 드는 노드가 있는 걸음마다 켜져
     레고·wolf 를 몸 전체 감쇠로 깸 -- s2). 모드 0 은 뒤집힘 걸음 수만 셈 (nflip).
  덜 풀린 걸음 (판정 커널 uf/cj = 1) 의 집합 S 노드는 원래 갱신 (--ue be) 이 먼저 (같은 후퇴 오일러).
근거: Newmark 계열 적분은 접촉 힘처럼 한 걸음 안에서 켜지고 꺼지는 힘을 걸음 양 끝 평균 (사다리꼴) 으로 넣어 충돌에서 에너지를 만들거나 되튐을 과장한다 --
  Deuflhard, Krause, Ertel 2008 (IJNME 73:1274, 'A contact-stabilized Newmark method for dynamical contact problems') 은 접촉 힘을 걸음 끝 시점에만 (암시적, 평균 없이)
  넣어 에너지가 늘지 않게 하고, Kane, Repetto, Ortiz, Marsden 1999 (CMAME 180:1) 도 충돌 걸음을 따로 다룸. H1·H2 는 '충돌 걸음 (노드) 은 1 차 암시 (후퇴 오일러)
  갱신 = 접촉 힘을 걸음 끝에만, 기억 없이', H3 은 '접촉 노드의 접촉 몫이 든 가속도를 다음 걸음 평균에 넣지 않음'.

훅 (끄면 = --hc 없으면 이 모듈을 안 부름): 실행기 runpy 순간 (다른 감싸개 설치 뒤, 가장 바깥) 에 ART.build_src·AORT._src 를 감싸 커널을 덧붙이고
  (hc_* + --ue_loc 갱신 커널 rt_ga_end_ufl·ao_stepend_ufl 의 복사본 rt_ga_end_hc·ao_stepend_hc = 노드 표시 hcm 이 서면 갱신 방식 hk), s.k 를 감싸
  걸음 끝 갱신 커널을 띄울 때 표시 커널을 먼저 띄우고 복사본으로 바꿔 띄움. 장벽 없는 씬 (ficus·bread) 은 rt_bar_q 를 안 불러 아무것도 안 바뀜 (원래 커널 그대로).
진단 (늘, 결과 불변): <out>/hc.jsonl -- step() 시도마다 걸음 수·충돌 걸음 수·새 접촉 노드 수·걸음 끝 접촉 노드 수·바꾼 노드 수 (합), K, t.
"""
import json
import os

import numpy as np
import torch

MODE = 0
ST = dict(on=False, said=False, out=".", log=None, nsrc=0, nhc=0, nbar=0)

HC_SRC = r'''
// ---- 절 CM58 (lib/anchorflow/fem_hhtc.py): 접촉 노드 판정 (걸음 시작·끝 장벽 받침 안) + 걸음 끝 갱신 바꿈 표시 ----
#ifdef E3W
#define HC_W(q, oc, o) E3W(q, o)
#else
#define HC_W(q, oc, o) BW[(oc)*27+(o)]
#endif
__global__ void hc_zero(const int* IP, int* ms, int* me, int* st) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0) st[2] = 0;                                                     // 이 걸음 새 접촉 노드 수
    if (n < IP[9]) { ms[n] = 0; me[n] = 0; }
}
__global__ void __launch_bounds__(128) hc_bar(const int* IP, const int* cb, const float* BW, const float* y, const float* U, float thinv,
                                              float n0, float n1, float n2, float pn, float dh, int* ms, int* me) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    float bm = AF_BX[q*4+3];
    if (!(bm > 0.f) || !AF_BACT[q]) return;
    int oc = q & 7; int ids[27]; qnodes(q, IP, cb, ids);
    float xs[3], xe[3];
    for (int a=0;a<3;a++) { xs[a] = AF_BX[q*4+a]/bm; xe[a] = xs[a]; }
    for (int o=0;o<27;o++) { float w = HC_W(q, oc, o); int id = ids[o];
        for (int a=0;a<3;a++) { float u = U[id*3+a]; xs[a] += w*u; xe[a] += w*(u + thinv*(y[id*3+a] - u)); } }
    float ds = xs[0]*n0 + xs[1]*n1 + xs[2]*n2 - pn, de = xe[0]*n0 + xe[1]*n1 + xe[2]*n2 - pn;
    bool s_ = ds < dh, e_ = de < dh;                                          // 장벽 받침 안 (rt_bar_q 가 힘을 거는 조건 d < d̂, 관통 d ≤ 0 포함)
    if (!s_ && !e_) return;
    for (int o=0;o<27;o++) if (HC_W(q, oc, o) != 0.f) { if (s_) ms[ids[o]] = 1; if (e_) me[ids[o]] = 1; }
}
// st: [0] 새 접촉 노드 합 [1] 걸음 끝 접촉 노드 합 [2] 이 걸음 새 접촉 노드 [3] 충돌 걸음 수 [4] 걸음 수 [5] 바꾼 노드 합
__global__ void hc_fin(const int* IP, const int* ms, const int* me, int mode, int* hm, int* st) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    int e = me[n], nw = e && !ms[n];
    if (e) atomicAdd(&st[1], 1);
    if (nw) { atomicAdd(&st[0], 1); atomicAdd(&st[2], 1); }
    int h = (mode == 3) ? e : ((mode == 2) ? nw : 0);
    hm[n] = h;
    if (h) atomicAdd(&st[5], 1);
}
__global__ void hc_all(const int* IP, int* st, int* hm) {                     // H1: 충돌 걸음이면 몸 전체
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || st[2] == 0) return;
    hm[n] = 1; atomicAdd(&st[5], 1);
}
// H4: 충돌 걸음 중 HHT 걸음 끝 속도 v1 = α (y − z0) 가 새 접촉 노드들의 법선 속도 합을 들어가는 쪽 (Σ v0·n < 0) 에서 나오는 쪽 (Σ v1·n > 0) 으로 뒤집는 걸음만 몸 전체
__global__ void hc_fz(double* fa) { fa[0] = 0.0; fa[1] = 0.0; }
__global__ void hc_flip(const int* IP, const int* ms, const int* me, const float* y, const float* z0, const float* vk, float al, float n0, float n1, float n2, double* fa) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || !(me[n] && !ms[n])) return;
    float v0 = vk[n*3]*n0 + vk[n*3+1]*n1 + vk[n*3+2]*n2;
    float v1 = al*((y[n*3] - z0[n*3])*n0 + (y[n*3+1] - z0[n*3+1])*n1 + (y[n*3+2] - z0[n*3+2])*n2);
    atomicAdd(&fa[0], (double)v0); atomicAdd(&fa[1], (double)v1);
}
__global__ void hc_all4(const int* IP, int* st, const double* fa, int* hm) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n == 0 && st[2] > 0 && fa[0] < 0.0 && fa[1] > 0.0) atomicAdd(&st[6], 1);   // 뒤집힌 충돌 걸음 수
    if (n >= IP[9] || !(st[2] > 0 && fa[0] < 0.0 && fa[1] > 0.0)) return;
    hm[n] = 1; atomicAdd(&st[5], 1);
}
__global__ void hc_cnt(int* st) { st[4] += 1; if (st[2] > 0) st[3] += 1; }
'''

SIG_OLD = "int um, const int* ulm) {"
SIG_NEW = "int um0, const int* ulm, const int* hcm, int hk) {\n    int um = um0;                                                              // 절 CM58: 노드마다 갱신 방식"
UL_LINE = "    uf_ = uf_ && ulm[n];"
HC_LINE = "    if (hcm[n] && !uf_) { uf_ = 1; um = hk; }                                    // 절 CM58: 표시 노드 = 후퇴 오일러 (hk 1) 또는 가속도 기억 0 (hk 2)\n"
PAIRS = (("rt_ga_end_ufl", "rt_ga_end_hc"), ("ao_stepend_ufl", "ao_stepend_hc"))


def hcify(src):
    """--ue_loc 갱신 커널 복사본 + 표시 커널을 extern "C" 끝에 (장벽 판 소스에만 표시 커널)."""
    add = []
    for nm, nh in PAIRS:
        key = "__global__ void " + nm + "("
        if key not in src:
            continue
        i = src.index(key)
        j = src.index("\n}\n", i) + 3
        k = src[i:j].replace(nm + "(", nh + "(", 1)
        assert k.count(SIG_OLD) == 1 and k.count(UL_LINE) == 1, ("절 CM58 hcify", nm)
        k = k.replace(SIG_OLD, SIG_NEW, 1)
        e = k.index("\n", k.index(UL_LINE)) + 1
        k = k[:e] + HC_LINE + k[e:]
        add.append(k)
        ST["nhc"] += 1
    if not add:
        return src
    if "__device__ float* AF_BX;" in src:
        add.append(HC_SRC)
        ST["nbar"] += 1
    ST["nsrc"] += 1
    s = src.rstrip()
    i = s.rfind("}")
    return s[:i] + "\n".join(add) + "\n}\n"


P = lambda t: np.uint64(0 if t is None else t.data_ptr())


def _buf(s, n):
    if getattr(s, "_hcst", None) is None:
        assert not torch.cuda.is_current_stream_capturing(), "절 CM58 버퍼는 데움 record 에서"
        s._hcst = torch.zeros(8, device=s.dev, dtype=torch.int32)
        s._hcfa = torch.zeros(2, device=s.dev, dtype=torch.float64)
    if getattr(s, "_hcm", None) is None or s._hcm.numel() < n:
        assert not torch.cuda.is_current_stream_capturing(), "절 CM58 노드 버퍼는 데움 record 에서"
        if getattr(s, "_hcm", None) is not None:                                         # 앞 그래프가 옛 포인터를 들고 있을 수 있어 버리지 않음
            s._hckeep = getattr(s, "_hckeep", []) + [s._hcm, s._hcms, s._hcme]
        m = int(n * 1.25) + 1024
        s._hcm = torch.zeros(m, device=s.dev, dtype=torch.int32)
        s._hcms = torch.zeros(m, device=s.dev, dtype=torch.int32)
        s._hcme = torch.zeros(m, device=s.dev, dtype=torch.int32)
        s._hcdum = torch.zeros(m, device=s.dev, dtype=torch.int32)                     # 모드 0 의 H4 셈용 버림 표시


def _wrap_k(cls, nm, nh):
    ok = cls.k

    def k(s, name, n, args, b=128):
        if name == "rt_bar_q" and len(args) >= 12:
            s._hcbq = (int(n), tuple(args))                                              # 마지막 장벽 호출의 표본점 수·기저·thinv·법선·평면·d̂
        if name == nm and getattr(s, "_hcbq", None) is not None:
            capN = int(n)
            _buf(s, capN)
            nq, ba = s._hcbq
            IP = args[0]
            y, U = args[5], args[7]                                                      # 두 실행기 모두 (IP, dt, gm, be, al, y, z0, U, ...)
            i32 = np.int32
            ok(s, "hc_zero", capN, (IP, P(s._hcms), P(s._hcme), P(s._hcst)), 256)
            ok(s, "hc_bar", nq, (IP, ba[1], ba[2], y, U, ba[6], ba[7], ba[8], ba[9], ba[10], ba[11], P(s._hcms), P(s._hcme)), 128)
            ok(s, "hc_fin", capN, (IP, P(s._hcms), P(s._hcme), i32(MODE), P(s._hcm), P(s._hcst)), 256)
            if MODE == 1:
                ok(s, "hc_all", capN, (IP, P(s._hcst), P(s._hcm)), 256)
            if MODE in (0, 4):                                                           # H4 (모드 0 은 셈만): HHT 가 새 접촉 노드 법선 속도를 뒤집는지
                ok(s, "hc_fz", 1, (P(s._hcfa),), 1)
                ok(s, "hc_flip", capN, (IP, P(s._hcms), P(s._hcme), y, args[6], args[8], args[4], ba[7], ba[8], ba[9], P(s._hcfa)), 256)
                ok(s, "hc_all4", capN, (IP, P(s._hcst), P(s._hcfa), P(s._hcm) if MODE == 4 else P(s._hcdum)), 256)
            ok(s, "hc_cnt", 1, (P(s._hcst),), 1)
            s._hcn = getattr(s, "_hcn", 0) + 1
            if MODE in (1, 2, 3, 4):
                return ok(s, nh, n, tuple(args) + (P(s._hcm), i32(2 if MODE == 3 else 1)), b)
        return ok(s, name, n, args, b)
    cls.k = k


def _wrap_step(cls):
    o_s = cls.step

    def step(s, t, *a, **kw):
        d = getattr(s, "_hcdep", 0)
        s._hcdep = d + 1
        try:
            r = o_s(s, t, *a, **kw)
        finally:
            s._hcdep = d
        if d == 0 and getattr(s, "_hcst", None) is not None and ST["log"] is not None:
            try:
                c = [int(q) for q in s._hcst.cpu().tolist()]
                s._hcst.zero_()
                ST["log"].write(json.dumps(dict(t=float(t), h=float(getattr(s, "h", 0.0) or 0.0), K=int(getattr(s, "Kcur", getattr(s, "K", 1)) or 1),
                                                new=c[0], eng=c[1], ncol=c[3], nstep=c[4], napp=c[5], nflip=c[6], nlaunch=int(getattr(s, "_hcn", 0)),
                                                ms=float(getattr(s, "last_ms", float("nan")) or float("nan")))) + "\n")
                ST["log"].flush()
            except Exception as e_:
                with open(os.path.join(ST["out"], "hc_err.txt"), "a") as f_:
                    f_.write(f"step 기록: {e_!r}\n")
        return r
    cls.step = step


def install(mode, out):
    """실행기 runpy 순간 (다른 감싸개 뒤) 에 부름. mode 0 = 진단만 (결과 불변), 1 H1 · 2 H2 · 3 H3."""
    global MODE
    MODE = int(mode)
    assert MODE in (0, 1, 2, 3, 4), MODE
    from . import fem_an_rt as ART
    from . import fem_an_rt_v0 as V0
    from . import fem_ao_rt as AORT
    ST.update(on=True, out=out)
    ST["log"] = open(os.path.join(out, "hc.jsonl"), "w")
    o_b, o_s = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: hcify(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: hcify(o_s(*a, **kw))
    _wrap_k(ART.RTSolver, "rt_ga_end_ufl", "rt_ga_end_hc")
    _wrap_k(AORT.AORTSolver, "ao_stepend_ufl", "ao_stepend_hc")
    for cls in (ART.RTSolver, V0.RTSolver):
        _wrap_step(cls)
    name = {0: "진단만 (결과 불변)", 1: "H1 충돌 걸음 몸 전체 후퇴 오일러", 2: "H2 새 접촉 노드 후퇴 오일러", 3: "H3 걸음 끝 접촉 노드 가속도 기억 0", 4: "H4 HHT 가 새 접촉 노드 법선 속도를 뒤집는 충돌 걸음만 몸 전체 후퇴 오일러"}[MODE]
    print(f"[절 CM58] hhtc 켬 -- 모드 {MODE}: {name}", flush=True)

    import atexit

    @atexit.register
    def _info():
        try:
            json.dump(dict(mode=MODE, nsrc=ST["nsrc"], nhc=ST["nhc"], nbar=ST["nbar"]), open(os.path.join(out, "hcinfo.json"), "w"))
        except Exception:
            pass
