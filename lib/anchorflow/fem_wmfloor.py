"""절 CM64 묶음 wmfloor (11-08 위임): 공통 후보 G' (F + A3 + P1X + H5 + Rf) 수박 바닥 뚫음·이탈 -- 고스트 질량만 얇은 구조 면까지 (M0·M5) + 진단 D6.

원인 (절 CM64.1, 기존 판 hhtc2 m5R_1~16 (H5) · hhtc m4R_1~16 (H4) 의 D2·D5·hc 기록, exe/wmfloor_an.py):
  이탈·바닥 뚫음 입자의 튐 프레임 (입자 속도가 한 프레임에 1 m/s 넘게 는 프레임) 66 개 모두 같은 꼴 -- 입자 몫이 가장 큰 노드가 **고정 아닌 (m ≥ mthr) 가벼운 노드**
  (질량 비 1e-3~5e-2 × 노드 질량 중앙, 문턱 1e-3 바로 위) 이고 그 노드가 한 걸음에 0.5~3 m 옮겨 끝 속도 3~112 m/s. 그 입자 칸은 거의 빈 칸 (칸 질량 비 0.05~0.08).
  P1X 의 3x3x3 열림이 빼는 '겉 부스러기 면' (절 CM55.2) 에 걸린 노드 = 벌점이 안 묶는 노드. H4·H5 몸 전체 후퇴 오일러는 그 프레임에 거의 안 켜짐 (H5 31 사건 중
  H4 뒤집힘 걸음 1, H4 판 35 사건 중 적용 1) -> 'H5 가 후퇴 오일러를 덜 걸어서' 는 아님.
  바닥 뚫음 = 이탈해 바닥에 놓인 입자 몇 개가 f75~f97 에 프레임당 0.2~2 mm 씩 바닥 밑으로 가라앉는 것 (깊은 낙하 아님, z 최소 −0.03~−0.05 m).

정의 (문턱·클램프 없음; 강성 벌점 면·γ̄·k_M 은 P1X 그대로):
  고스트 질량 (Stoter 외 2023 식 (27)~(29), 절 CM51 (1)) 의 면 집합만 넓힘 -- 강성 벌점은 P1X 면 (두 칸 모두 두꺼운 칸, 구속 노드 면 빼기) 그대로,
  고스트 질량은 아래 면에도 (같은 k_M = γ̄ ρ̄ h³, 같은 커널 = fem_ghost6 gq_* 를 강성 0 으로 한 번 더):
    M0 (AF_WF_MM=0): 절 CM47 후보 면 전부 (축 이웃 두 칸 활성, 하나 이상 경계 칸) -- 절 CM53 방식 0 의 면.
    M5 (AF_WF_MM=5): 후보 면 중 한쪽 이상이 두꺼운 칸인 면 (절 CM55 방식 5 의 면 = 덩어리에 붙은 겉 부스러기 면; 덩어리에서 두 칸 넘게 떨어진 얇은 구조 면은 뺌).
  구속 노드 면 빼기 (P1X @g4x 1) 는 더한 면에도 같음. 더한 면 = (위 면) \ (P1X 강성 면) 이라 P1X 면의 질량 몫은 두 번 안 들어감.
  까닭: 고스트 질량은 관성 예측에서 벗어난 변위 (y − ŷ) 의 면 법선 2 차 미분 점프만 벌함 = 받침이 작은 노드의 가속도를 이웃 노드 가속도의 매끈한 연장에 묶음 (문헌의
  목적이 '작은 칸의 질량 결손'). 정적 평형·강체·균일 변형 운동에는 힘 0 (G 가 아핀 장에 0) 이라 강성 벌점처럼 얇은 구조를 굳히지 않음 -- 절 CM51 가르기에서
  ficus 주기에 질량 몫 영향 없음 (두꺼운 면에서), 절 CM53.2 에서 수박 (a) 강성만 -> (b) + 질량 튐 3 -> 1.
진단 (결과 불변):
  wf.jsonl -- 프레임마다 더한 질량 면 수 (구속 빼기 뒤).
  D6 (AF_WF_D6=1) -- 프레임 끝 바닥 밑 입자 (d = x·n − p_n < 0) 수·바닥 밑 한 칸 넘는 입자, 그 입자들의 이어 받은 기준 높이 h_p (절 CM49 R2) 와 실제 d 의 차 (가상 높이),
     가장 깊은 입자 64 개의 27 노드 고정 몫 (pm 2) 과 최소 질량 비, 처음 바닥 밑 한 칸을 넘은 입자의 추적 (프레임별 d·h_p).
  검사 (AF_WF_CHK=프레임 목록, 그래프 밖): 면 수 |M면| = |P1X 면| + |더한 면|, 임의 벡터 dy 에 대해 G(M면) dy = G(P1X 면) dy + G(더한 면) dy.
"""
import json
import os

import numpy as np
import torch

MM = int(os.environ.get("AF_WF_MM", "-1"))                                                # -1 = 더한 면 없음 (진단만)
D6 = int(os.environ.get("AF_WF_D6", "0"))
CHK = {int(q) for q in os.environ.get("AF_WF_CHK", "").split(",") if q.strip()}
ST = dict(on=False, out=".", log=None, d6=None, nlaunch=0, track=None, said=False)

WF_SRC = r'''
// ---- 절 CM64 wmfloor: 고스트 질량만 더할 면 (P1X 강성 면 밖) ----
// mm 0: 후보 면 (두 칸 활성, 하나 이상 경계 칸) 중 두 칸 모두 두꺼운 칸이 아닌 면, mm 5: 그중 한쪽 이상 두꺼운 칸인 면. thk 비트 0 = 3x3x3 두꺼운 칸 (AF_G4_THIN 1).
__global__ void gq_wfface(const int* IP, const int* cb, const int* cidx, const unsigned char* cut, const unsigned char* thk, unsigned char* fm, int mm, int* cnt) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c == 0) atomicAdd(&cnt[3], 1);
    if (IP[13] || c >= IP[11]) return;
    int n[3] = {IP[6]-2, IP[7]-2, IP[8]-2};
    int m = 0, nk = 0;
    for (int a=0;a<3;a++) {
        int x[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x[a] += 1;
        if (x[a] >= n[a]) continue;
        int c2 = cidx[(x[0]*n[1] + x[1])*n[2] + x[2]];
        if (c2 < 0 || !(cut[c] || cut[c2])) continue;
        bool t1 = (thk[c] & 1), t2 = (thk[c2] & 1);
        if (t1 && t2) continue;                                                   // P1X 강성 면 (강성 + 질량은 ghost6 이 이미)
        bool on = (mm == 0) ? true : (t1 || t2);
        if (on) { m |= (1 << a); nk++; }
    }
    fm[c] = (unsigned char)m;
    if (nk) atomicAdd(&cnt[0], nk);
}
__global__ void gq_wfzero(int* cnt) { cnt[0] = 0; cnt[1] = 0; cnt[2] = 0; }
__global__ void gq_wfpop(const int* IP, const unsigned char* fm, int* cnt, int slot) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    int k = (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1);
    if (k) atomicAdd(&cnt[slot], k);
}
'''


def _append(src):
    if "gq_wfface(" in src:
        return src
    assert "gq_force(" in src, "wmfloor: ghost6 커널 (gq_force) 이 먼저 붙어야 함"
    i = src.rstrip().rfind("}")
    return src[:i] + WF_SRC + src[i:]


def install(out):
    """실행기 runpy 순간 (ghost6·hhtc2 뒤, 가장 바깥) 에 부름."""
    from . import fem_an_rt as ART
    from . import fem_an_rt_v0 as V0
    from . import fem_ao_rt as AORT
    from . import fem_ghost6 as G3
    assert G3.ST["on"], "wmfloor: ghost6 (P1X) 위에서만"
    if MM >= 0:
        assert G3.THIN == 1 and G3.MASS == 1 and G3.WMODE == 0 and not G3.CONS, "wmfloor: P1X (@g4t 1, 질량 켬, 면 가중 없음, 증분 판) 위에서만"
        assert MM in (0, 5), MM
    ST.update(on=True, out=out)
    ST["log"] = open(os.path.join(out, "wf.jsonl"), "w")
    o_b, o_s = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: _append(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: _append(o_s(*a, **kw))
    P, f32 = G3.P, np.float32

    def _alloc(s):
        cap = int(s.capQc)
        if getattr(s, "wffm", None) is None or s.wffm.numel() < cap:
            s.wffm = torch.zeros(cap, device=s.dev, dtype=torch.uint8)
        if getattr(s, "wfcnt", None) is None:
            s.wfcnt = torch.zeros(4, device=s.dev, dtype=torch.int32)
            s.wfGQ = torch.zeros(8, device=s.dev, dtype=torch.float64)

    def hook(s, ok, name, args):
        if MM < 0 or getattr(s, "gpcnt", None) is None:
            return
        if name == "rt_gauss":
            _alloc(s)
            cq = int(s.capQc)
            ok(s, "gq_wfzero", 1, (P(s.wfcnt),), 1)
            ok(s, "gq_wfface", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3cut), P(s.g3thk), P(s.wffm), np.int32(MM), P(s.wfcnt)), 128)
            return
        if name == "rt_bc_box":
            if G3.FIXX and getattr(s, "wffm", None) is not None:
                ok(s, "gq_fix", int(s.capQc), (P(s.IP), P(s.cb), args[4], P(s.wffm), P(None)), 128)
            return
        if getattr(s, "wffm", None) is None:
            return
        if bool(getattr(s, "det", False)):
            return
        if name in ("en_phi_n",):                                                            # Φ: 질량 몫만 (it2 = 이 커널 인자)
            cmp = float(s._g3kM) * float(args[2])
            if cmp != 0.0:
                ok(s, "gq_phi", int(s.capQc), (P(s.IP), P(s.cb), P(s.wffm), P(None), f32(0.0), f32(cmp), args[3], args[4], P(None), args[5], args[6]), 128)
                ST["nlaunch"] += 1
            return
        cm = G3.cm_of(s)
        if cm == 0.0:                                                                        # 걸음 밖: 질량 몫 없음 (ghost6 와 같음)
            return
        nq = int(s.capQc)
        G = (P(s.IP), P(s.cb), P(s.wffm), P(None))
        YH = P(s.yh)
        if name in G3.ELEM:
            if name == "rt_elem":
                mode = int(args[8]); y, E, f_, B = args[5], args[9], args[10], args[13]
                ok(s, "gq_force", nq, G + (f32(0.0), f32(cm), y, YH, P(None), f_ if mode & 1 else P(None), B if mode & 8 else P(None), E if mode & 4 else P(None)), 128)
            elif name == "aq_elem":
                ok(s, "gq_force", nq, G + (f32(0.0), f32(cm), args[5], YH, P(None), args[10], args[13], P(None)), 128)
            else:
                mode = int(args[9])
                ok(s, "gq_force", nq, G + (f32(0.0), f32(cm), args[5], YH, P(None), args[10] if mode & 1 else P(None), args[13] if mode & 8 else P(None), P(None)), 128)
        elif name in G3.MV:
            ok(s, "gq_mv", nq, G + (f32(cm), args[6], args[7]), 128)
        elif name == "rt_bd":
            ok(s, "gq_force", nq, G + (f32(0.0), f32(cm), P(s.y), YH, P(None), P(None), args[6], P(None)), 128)
        elif name == "rt_elem_lsN":
            yh, eta, dl, sk, pm, n0, n1, n2, SC, NC, i0, gate = args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[15], args[16], args[17], args[18]
            ok(s, "gq_ls", nq, G + (yh, P(None), eta, dl, sk, pm, n0, n1, n2, P(s.wfGQ), gate), 128)
            ok(s, "gq_lsfin", 1, (SC, NC, i0, gate, P(s.wfGQ), f32(0.0), f32(cm)), 1)
        else:
            return
        ST["nlaunch"] += 1

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            r = _ok(s, name, n, args, b)
            if getattr(s, "_wfin", False) or name.startswith("gq_") or name.startswith("gp_"):
                return r
            s._wfin = True
            try:
                hook(s, _ok, name, args)
            finally:
                s._wfin = False
            return r
        cls.k = k

    for cls in (ART.RTSolver, V0.RTSolver):
        wrap_k(cls)

    # ---- 프레임 끝 기록 (가장 바깥 쪼개기 함수 뒤, 그래프 밖)
    from . import fem_wmrx as WRX
    dep = dict(d=0)

    def outer(fn):
        def w(s, t, h, *a, **kw):
            dep["d"] += 1
            try:
                res = fn(s, t, h, *a, **kw)
            finally:
                dep["d"] -= 1
            if dep["d"] == 0:
                fi = int(round(t / h)) + 1
                try:
                    post(s, fi)
                except Exception:
                    import traceback
                    with open(os.path.join(out, "wf_err.txt"), "a") as f_:
                        f_.write(traceback.format_exc()[-3000:] + "\n")
            return res
        return w
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    nm = {-1: "더한 면 없음 (진단만)", 0: "M0 고스트 질량 면 = 후보 면 전부", 5: "M5 고스트 질량 면 = 한쪽 이상 두꺼운 칸인 후보 면"}[MM]
    print(f"[절 CM64] wmfloor 켬 -- {nm}; 강성 면 P1X 그대로; D6 {D6}; 검사 프레임 {sorted(CHK)}", flush=True)


@torch.no_grad()
def post(s, fi):
    rec = dict(f=fi)
    if getattr(s, "wfcnt", None) is not None and MM >= 0:
        from . import fem_ghost6 as G3
        c = s.wfcnt
        # 구속 빼기 뒤 더한 면 수 (그래프 밖에서 다시 셈)
        tmp = torch.zeros(4, device=s.dev, dtype=torch.int32)
        s._wfin = True
        try:
            s.k("gq_wfpop", int(s.capQc), (G3.P(s.IP), G3.P(s.wffm), G3.P(tmp), np.int32(0)), 128)
            s.k("gq_wfpop", int(s.capQc), (G3.P(s.IP), G3.P(s.g3fm), G3.P(tmp), np.int32(1)), 128)
        finally:
            s._wfin = False
        torch.cuda.synchronize()
        t_ = tmp.cpu().tolist()
        rec.update(nadd=int(t_[0]), np1x=int(t_[1]), nadd_build=int(c[0].item()), nbuild=int(c[3].item()), nl=ST["nlaunch"])
        if fi in CHK:
            rec["chk"] = check(s)
    if D6:
        rec["d6"] = d6(s, fi)
    ST["log"].write(json.dumps(rec) + "\n")
    ST["log"].flush()


@torch.no_grad()
def check(s):
    """그래프 밖: (1) |M 면| = |P1X 면| + |더한 면| (2) G(M 면) dy = G(P1X 면) dy + G(더한 면) dy (kk = 1, 임의 dy)."""
    from . import fem_ghost6 as G3
    _ok = lambda s_, nm, n, a, b: s_.k(nm, n, a, b)
    s._wfin = True
    s._gpin = True
    try:
        if getattr(s, "g3dummy", None) is None:
            s.g3dummy = torch.zeros(4, device=s.dev, dtype=torch.int32)
        full = torch.zeros_like(s.g3fm)
        G3.build_mask(s, _ok, 0 if MM == 0 else 5, full, None)
        if G3.FIXX and getattr(s, "fixd", None) is not None:
            G3.apply_fix(s, _ok, full, None)
        torch.cuda.synchronize()
        cnt = torch.zeros(4, device=s.dev, dtype=torch.int32)
        for i_, fm in enumerate((full, s.g3fm, s.wffm)):
            s.k("gq_wfpop", int(s.capQc), (G3.P(s.IP), G3.P(fm), G3.P(cnt), np.int32(i_)), 128)
        torch.cuda.synchronize()
        g = torch.Generator(device=s.dev).manual_seed(7)
        dy = torch.randn(s.y.numel(), device=s.dev, generator=g, dtype=torch.float32) * 1e-3
        outs = []
        for fm in (full, s.g3fm, s.wffm):
            o = torch.zeros_like(dy)
            s.k("gq_mv", int(s.capQc), (G3.P(s.IP), G3.P(s.cb), G3.P(fm), G3.P(None), np.float32(1.0), G3.P(dy), G3.P(o)), 128)
            outs.append(o)
        torch.cuda.synchronize()
        a, b = outs[0].double(), (outs[1] + outs[2]).double()
        rel = float((a - b).norm() / a.norm().clamp_min(1e-30))
        # ghost6 면 표지·두꺼움 되돌림 (방식 1 + 구속 빼기)
        G3.build_mask(s, _ok, G3.THIN, s.g3fm, None)
        if G3.FIXX and getattr(s, "fixd", None) is not None:
            G3.apply_fix(s, _ok, s.g3fm, None)
        torch.cuda.synchronize()
        c = cnt.cpu().tolist()
        return dict(nfull=c[0], np1x=c[1], nadd=c[2], ok_cnt=bool(c[0] == c[1] + c[2]), mv_rel=rel, mv_norm=float(a.norm()))
    finally:
        s._wfin = False
        s._gpin = False


def _bw(fx):
    return torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], dim=-1)


@torch.no_grad()
def d6(s, fi):
    from . import fem_escape4 as E4
    nr, pn, dh = E4.plane(s)
    N = int(getattr(s, "N", s.x.shape[0]))
    x = s.x[:N].double()
    d = x @ torch.tensor(nr, device=x.device, dtype=torch.float64) - pn
    ok = torch.isfinite(d)
    cell = float(s.A[:, 0].norm())
    b0 = ok & (d < 0)
    b1 = ok & (d < -cell)
    r = dict(nb0=int(b0.sum()), nb1=int(b1.sum()), dmin=float(d[ok].min()), dh=dh, cell=cell)
    hp = getattr(s, "e5hr", None)
    if hp is not None and bool(b0.any()):
        h = hp[:N].double()
        hb = h[b0]
        db = d[b0]
        r.update(hp_pos=int((hb > 0).sum()), off_max=float((hb - db).max()), off_med=float((hb - db).median()), hp_min=float(hb.min()), hp_max=float(hb.max()))
    if bool(b1.any()):
        idx = torch.nonzero(b1).squeeze(1)
        idx = idx[torch.argsort(d[idx])[:64]]
        IP = s.IP.cpu()
        nb1, nb2, nn = int(IP[7]), int(IP[8]), int(IP[9])
        pb = s.Np.reshape(-1, 3)[idx].long()
        fx = s.fx.reshape(-1, 3)[idx].float()
        w = _bw(fx)
        o = torch.arange(27, device=fx.device)
        i, j, k = o // 9, (o // 3) % 3, o % 3
        ww = w[:, 0, i] * w[:, 1, j] * w[:, 2, k]
        nid = (((pb[:, 0:1] + i) * nb1 + (pb[:, 1:2] + j)) * nb2 + (pb[:, 2:3] + k)).clamp(0, nn - 1)
        mn = s.mn[:nn].float()
        med = float(mn[mn > 0].median())
        pin = (s.pm[:nn] == 2)[nid].float()
        big = ww > 1e-3
        r.update(deep=[dict(p=int(p_), d=round(float(d[p_]), 5), hp=(round(float(hp[p_]), 5) if hp is not None else None),
                            wpin=round(float((ww[q] * pin[q]).sum()), 3), mnmin=float((mn[nid[q]][big[q]] / med).min()) if bool(big[q].any()) else None,
                            v=[round(float(t_), 3) for t_ in s.v[p_].tolist()]) for q, p_ in enumerate(idx.tolist()[:8])],
                 wpin_med=float((ww * pin).sum(1).median()))
        if ST["track"] is None:
            ST["track"] = idx[:4].tolist()
    if ST["track"] is not None:
        tr = torch.tensor(ST["track"], device=s.x.device)
        r["track"] = [dict(p=int(p_), d=round(float(d[p_]), 5), hp=(round(float(hp[p_]), 5) if hp is not None else None), vz=round(float(s.v[p_, 2]), 4)) for p_ in tr.tolist()]
    return r
