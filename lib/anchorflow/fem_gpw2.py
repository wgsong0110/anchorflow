"""절 CM99 묶음 gpw2 (사용자 11-10 지시): 절 CM95 면 가중 w_F = 1 − φ_min 은 그대로 두고 칸 점유 φ 의 정의만 바꿈 (연속 정의, 새 문턱·클램프 없음).

절 CM95 의 φ = min(1, V_c / h³) (V_c = 칸 입자 부피 합) 는 wolf 모래 덩어리 안쪽도 0.3 안팎이라 w_F 가 0.65~0.73 으로 남았다 (입자 부피 몫 ≠ 기하 채움).
두 정의 (AF_GPHI / --gphi):
  1 = (G) 기하 채움: φ_G(c) = (1/8) Σ_{o=1..8} 1[칸 c 의 팔분 o 에 입자가 하나라도 있음].
      팔분 = 칸 중심에서 각 축을 반으로 나눈 8 부분 칸 (rt_p2g 의 팔분 번호 oc = 4·[fx ≥ 1] + 2·[fy ≥ 1] + [fz ≥ 1], 가우스 점 q = 8c + oc 와 같은 분할).
      '팔분에 입자가 있음' = 그 팔분 가우스 점 누적 부피 A[q·28 + 27] (= 그 팔분 입자 부피 합) > 0. 입자 부피 크기와 무관하게 입자가 칸의 어느 부분을 실제로 덮는지 (8 단계 몫).
      근거: 잘린 칸 (cut cell) 의 잘린 부피 몫 κ_K = |K ∩ Ω| / |K| (Frei 외 2024 2.3 절; Burman & Hansbo 2012) 은 '물체가 칸을 차지하는 기하 몫' 이다.
      입자 표현에서 Ω 를 입자가 있는 부분 칸들의 합집합으로 근사하면 그 몫이 φ_G (부분 칸 해상도 h/2). 입자 부피 합 / 칸 부피는 입자 사이 빈틈 (모래 충전율) 까지
      결손으로 세어 안쪽 칸도 '덜 참' 이 된다 -- φ_G 는 입자 사이 빈틈을 결손으로 세지 않는다 (팔분당 평균 입자 수가 1 보다 충분히 크면 안쪽 칸 φ_G → 1).
  2 = (N) 첫 프레임 정규화: φ_N(c) = min(1, φ(c) / φ_0),  φ_0 = 그 씬 첫 프레임 (첫 rt_gauss) 에 찬 칸 (φ > 0) 들의 φ 중앙값 (실행 중 고정).
      덩어리 안쪽 (첫 프레임의 전형적 칸) 이 1 이 되게 φ 의 단위를 '그 재질의 충전 상태' 로 바꿈 (모래 충전율 ~0.3 이면 0.3 을 '꽉 참' 으로).
      min(1, ·) 은 절 CM55·CM95 φ = min(1, V_c/h³) 의 그것 (몫은 1 을 넘지 않는다는 정의) 을 정규화된 몫에 그대로 둔 것 -- 새 값은 없음.
      (이것 없이는 φ_0 보다 빽빽한 칸 (첫 프레임 칸의 절반) 에서 w_F = 1 − φ_min < 0 = 음의 벌점이 된다.) 몫이 1 에 닿는 칸 비율은 진단에 적음.
가중은 절 CM95 그대로 (fem_gpw: ghost6 방식 5 경로는 gq5_w, lightg3 켜진 씬은 gq_gpw) -- 둘 다 s.g5phi 를 읽으므로 이 모듈은 rt_gauss (+ ghost6 의 gq5_cell) 바로 뒤에
s.g5phi 만 새 φ 로 덮고, 방식 5 경로면 gq5_w 를 새 φ 로 한 번 더 부름 (같은 커널·같은 인자). g5phi 는 면 가중 밖에서 안 읽힘 (ghost6 gq5_w·gq_gpw·진단뿐).
gpw 가 씬에 안 쓰이는 판 (모드 2 의 κ ≠ 0 씬) 은 아무것도 안 함 (H5 그대로).
진단 (AF_GPW_DG=1, 결과 불변): gphi.jsonl 프레임 끝 = 실행 φ 대 torch 참값 최대 차 (phierr), 안쪽 칸 (gq_cut 이 아닌 칸 = 26 이웃 모두 찬 칸) φ 분위 (새 정의·절 CM55 정의),
  P1X 면 중 안쪽 칸에 닿는 면 (덩어리 안쪽 쪽 면) 의 w_F 분위·평균·수, 모든 P1X 면 w_F 분위 (절 CM95 gpw.jsonl 과 같은 값), (N) 은 몫이 1 에 닿은 찬 칸 비율.
"""
import json
import os

import numpy as np
import torch

PHI = int(os.environ.get("AF_GPHI", "0"))
ST = dict(phi=PHI, phi0=None, nk=0, nw=0, nfr=0, out=".", err=None)

SRC = r'''
// ---- 절 CM99 gpw2: 칸 점유 φ 다시 정의 (lib/anchorflow/fem_gpw2.py) -- (G) 팔분 기하 채움, (N) 첫 프레임 정규화
__global__ void gq_phig(const int* IP, const float* A, float* phi) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int k = 0;
    for (int o=0;o<8;o++) k += (A[((size_t)c*8 + o)*28 + 27] > 0.f) ? 1 : 0;
    phi[c] = 0.125f*(float)k;
}
__global__ void gq_phin(const int* IP, float inv0, float* phi) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    phi[c] = fminf(phi[c]*inv0, 1.f);
}
'''


def _append(src):
    if "gq_phig(" in src:
        return src
    i = src.rstrip().rfind("}")
    return src[:i] + SRC + "\n}\n"


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _info():
    with open(os.path.join(ST["out"], "gphiinfo.json"), "w") as f_:
        json.dump(ST, f_, ensure_ascii=False, default=str)


def _q(v):
    if v.numel() == 0:
        return None
    return [round(float(x), 4) for x in torch.quantile(v.float()[:1000000], torch.tensor([0.1, 0.5, 0.9], device=v.device)).tolist()]


def ref(s):
    """torch 참값: 새 φ (f64) 와 절 CM55 φ."""
    from . import fem_ghost6 as G6
    nc = int(s.IP[11])
    ph55 = G6.ref_phi(s)[0]
    if ST["phi"] == 1:
        a = s.Aacc.reshape(-1)[:nc * 8 * 28].view(nc * 8, 28)[:, 27]
        return (a > 0).view(nc, 8).double().mean(1), ph55
    return torch.clamp(ph55.float().double() / float(ST["phi0"]), max=1.0), ph55


def diag(s, fi):
    """절 CM95 diag 뒤 (프레임 끝): 새 φ 참값 대조·안쪽 칸·안쪽 면 분포."""
    from . import fem_gpw as GPW
    from . import fem_ghost6 as G6
    nc = int(s.IP[11])
    php, ph55 = ref(s)
    run = s.g5phi[:nc].double()
    rec = dict(f=fi, nc=nc, phierr=float((run - php).abs().max()) if nc else 0.0)
    inn = s.g3cut[:nc] == 0
    rec.update(ninner=int(inn.sum()), phi_in=_q(run[inn]), phi55_in=_q(ph55[inn]), phi_all=_q(run), phi55_all=_q(ph55))
    if ST["phi"] == 2:
        rec.update(phi0=ST["phi0"], cap1=round(float((ph55 / float(ST["phi0"]) >= 1.0).double().mean()), 4) if nc else None)
    keys = G6.mask_keys(s.g3fm, nc)
    if GPW.ST["path"] == "mul":
        keys = keys[s.l3id[keys] > 0.5]
    w = s.g5w[keys].double()
    c = keys // 3
    a = keys - 3 * c
    cb = s.cb.reshape(-1)[:nc * 3].long().view(nc, 3)
    n0, n1, n2 = int(s.IP[6]) - 2, int(s.IP[7]) - 2, int(s.IP[8]) - 2
    x2 = cb[c].clone()
    x2[torch.arange(c.numel(), device=c.device), a] += 1
    inb = (x2[:, 0] < n0) & (x2[:, 1] < n1) & (x2[:, 2] < n2)
    x2 = torch.where(inb[:, None], x2, torch.zeros_like(x2))
    c2 = torch.where(inb, s.cidx.reshape(-1).long()[(x2[:, 0] * n1 + x2[:, 1]) * n2 + x2[:, 2]], torch.full_like(c, -1))
    fin = inn[c] | ((c2 >= 0) & inn[c2.clamp(min=0)])                                      # 덩어리 안쪽 쪽 면 = 안쪽 칸 (26 이웃 모두 찬 칸) 에 닿는 P1X 면
    wi = w[fin]
    rec.update(nface=int(w.numel()), wmean=round(float(w.mean()), 5) if w.numel() else None, wq=_q(w),
               nface_in=int(wi.numel()), wmean_in=round(float(wi.mean()), 5) if wi.numel() else None, wq_in=_q(wi),
               whalf_in=round(float((wi < 0.5).double().mean()), 4) if wi.numel() else None)
    ST["log"].write(json.dumps(rec) + "\n")
    ST["log"].flush()


def install(out, phi=None):
    """gpw2_wrap: fem_gpw.install 바로 뒤 (가장 바깥 감싸개) 에 부름."""
    from . import fem_an_rt as ART
    from . import fem_an_rt_v0 as V0
    from . import fem_ao_rt as AORT
    from . import fem_ghost6 as G6
    from . import fem_gpw as GPW
    if phi is not None:
        ST["phi"] = int(phi)
    assert ST["phi"] in (1, 2), ("gpw2: φ 정의는 1 (G 기하 채움) · 2 (N 첫 프레임 정규화)", ST["phi"])
    assert GPW.ST["mode"] in (1, 2), "gpw2: fem_gpw (절 CM95 면 가중) 가 먼저 켜져 있어야 함"
    ST["out"] = out
    DG = int(GPW.DG)
    ST["log"] = open(os.path.join(out, "gphi.jsonl"), "w") if DG else None
    o_b, o_s = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: _append(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: _append(o_s(*a, **kw))

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            r = _ok(s, name, n, args, b)
            if name != "rt_gauss" or not GPW.ST["dec"] or getattr(s, "g5phi", None) is None:
                return r
            cq = int(s.capQc)
            if ST["phi"] == 1:
                _ok(s, "gq_phig", cq, (P(s.IP), args[1], P(s.g5phi)), 128)
            else:
                if ST["phi0"] is None:                                                         # 첫 rt_gauss (첫 프레임, 그래프 밖): 찬 칸 φ 중앙값
                    assert not torch.cuda.is_current_stream_capturing(), "gpw2 (N): 첫 프레임 φ_0 를 그래프 캡처 안에서 잴 수 없음"
                    torch.cuda.synchronize()
                    nc = int(s.IP[11])
                    if int(s.IP[13]) != 0 or nc == 0:
                        return r                                                               # 넘침 (다시 할당 뒤 다시 옴)
                    v = s.g5phi[:nc]
                    v = v[v > 0]
                    ST["phi0"] = float(torch.median(v.double()))                                # 낮은 쪽 중앙 (짝수 개면 두 가운데 중 작은 것)
                    ST["phi0_n"] = int(v.numel())
                    ST["phi0_q"] = _q(v)
                    print(f"[절 CM99 gpw2] (N) φ_0 = 첫 프레임 찬 칸 {v.numel()} 개 φ 중앙값 {ST['phi0']:.5g} (10/50/90 % {ST['phi0_q']})", flush=True)
                    _info()
                _ok(s, "gq_phin", cq, (P(s.IP), np.float32(1.0 / ST["phi0"]), P(s.g5phi)), 128)
            ST["nk"] += 1
            if GPW.ST["path"] == "wmode5":                                                     # ghost6 가 rt_gauss 훅에서 옛 φ 로 쓴 가중을 새 φ 로 다시 (같은 커널·인자)
                assert G6.WMODE == 5, ("gpw2: 방식 5 경로인데 WMODE", G6.WMODE)
                _ok(s, "gq5_w", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3fm), P(s.g5phi), np.int32(G6.WMODE), np.float32(G6.WMAX), P(s.g5w)), 128)
                ST["nw"] += 1
            if ST["nk"] <= 2:
                _info()
            return r
        cls.k = k

    for cls in (ART.RTSolver, V0.RTSolver):
        wrap_k(cls)

    o_diag = GPW.diag

    def diag2(s, fi):
        o_diag(s, fi)
        ST["nfr"] += 1
        try:
            if ST["log"] is not None:
                diag(s, fi)
            if ST["nfr"] <= 2 or ST["nfr"] % 10 == 0:
                _info()
        except Exception:
            import traceback
            with open(os.path.join(out, "gphi_err.txt"), "a") as f_:
                f_.write(f"f{fi}\n" + traceback.format_exc()[-3000:] + "\n")
    GPW.diag = diag2
    _info()
    print(f"[절 CM99] gpw2 켬 -- 칸 점유 φ = {'(G) 팔분 기하 채움' if ST['phi'] == 1 else '(N) 첫 프레임 정규화 min(1, φ/φ_0)'}, 면 가중 w_F = 1 − φ_min (절 CM95), 진단 {DG}", flush=True)
