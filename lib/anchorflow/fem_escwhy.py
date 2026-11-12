"""절 CM126 묶음 escwhy (조정자 11-12, 진단 -- 결과 불변): wolf 입자 이탈 (튀는 입자) 불안정의 원인을 걸음별 기록으로 가르기.

프레임 창 [F0, F1] (env ESC_F0·ESC_F1, 1 부터) 의 프레임마다 <out>/esc/fNNN.npz 에 남김 (그래프 안 사본, 풀이 버퍼는 읽기만):
  프레임: IP (상자), A·Ai, 노드 질량 mn·mqn·운동량 mv (P2G 결과), 가우스 점 F_q (rt_gauss 직후)·V_q·m_q·xq0·장벽 표본점 bx, 칸 cb,
          입자 (원래 순서) 프레임 시작·끝 x·v·F·a_p (perm 으로 되돌림).
  걸음 k (HHT 걸음 시작 rt_ga_step0 뒤): U0·v0·a0 (걸음 시작), Ũ·z0;
          걸음 끝 갱신 (rt_ga_end / _uf / _ufl -- 안쪽 감싸개가 무엇으로 바꾸든) 앞: 뉴턴 해 y, ŷ, 바닥 sk·pm, 고착 몫 an, 유효 질량 meff,
            마지막 반복 시작 잔차 g0 (접기 전), 힘 f (f64), 장벽 노드 기울기·헤시안 bGn·bKn, 확장 묶음 w8b (−1 = 내부·고립·빈 노드), 구속 인자 kb, 갱신 바꿈 집합 ulm;
          뒤: U1·v1; ao_commit 뒤: a1 (구속 노드 0 뒤);
          ao_commit 앞: 가우스 점 시험 X = (I + ∇U) F_q 의 특이값·DP 상태 (0 탄성 1 꼬리 2 꼭짓점)·det X (esc_q 커널), F_q (커밋 전).
노드 배열은 mn > 0 노드만 (번호 nid 같이), 가우스 점은 [:nq].
훅: 실행기 runpy 순간 (가장 바깥). 버퍼 합이 ESC_MAXB (기본 8e9 바이트) 를 넘는 용량이면 그 할당에서는 기록 안 함 (기록 안 한 프레임 표시).
"""
import json
import os

import numpy as np
import torch

ST = dict(out=".", nsrc=0, nalloc=0, nsave=0, nskip=0, err=None, F0=1, F1=10 ** 9, on_alloc=None)
MARK = "// ---- 절 CM126 escwhy"
HEAD = "__global__ void ao_commit("
ESC_SRC = r"""
// ---- 절 CM126 escwhy: 가우스 점 시험 X (커밋 앞) 의 특이값·DP 상태·det (읽기만) ----
__global__ void esc_q(const int* IP, const int* cb, const float* BG, const float* Fq, const float* U, float cdp, float* out) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    for (int o_=0;o_<27;o_++) { float g[3]; gqn(F, BG + (oc*27+o_)*3, g); int id = ids[o_];
        for (int a=0;a<3;a++) { float ua = U[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; } }
    float Uu[9], S[3], V[9];
    ssvdf(X, Uu, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    float dX = X[0]*(X[4]*X[8]-X[5]*X[7]) - X[1]*(X[3]*X[8]-X[5]*X[6]) + X[2]*(X[3]*X[7]-X[4]*X[6]);
    float* o = out + (size_t)q*5;
    o[0] = S[0]; o[1] = S[1]; o[2] = S[2]; o[3] = (float)st; o[4] = dX;
}
"""
ENDK = ("rt_ga_end", "rt_ga_end_uf", "rt_ga_end_ufl")


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def patch(code):
    if MARK in code or HEAD not in code or "ao_dpf(" not in code:
        return code
    i = code.index(HEAD)
    ST["nsrc"] += 1
    return code[:i] + MARK + "\n" + ESC_SRC + code[i:]


def _dump():
    try:
        with open(os.path.join(ST["out"], "escinfo.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


NODE3 = ("U0", "V0", "A0", "Ut", "Z0", "Y", "YH", "G0", "U1", "V1", "A1")
NODE1 = ("SK", "AN", "ME", "BG", "BK", "KB")
NODEI = ("PM", "W8", "UL")


def install(out):
    from . import fem_an_rt as ART
    ST.update(out=out, F0=int(os.environ.get("ESC_F0", "1")), F1=int(os.environ.get("ESC_F1", str(10 ** 9))))
    maxb = float(os.environ.get("ESC_MAXB", "8e9"))
    os.makedirs(os.path.join(out, "esc"), exist_ok=True)
    cpm = ART.cp
    orm = cpm.RawModule

    def RawModule(*a, **kw):
        if "code" in kw:
            kw["code"] = patch(kw["code"])
        elif a:
            a = (patch(a[0]),) + tuple(a[1:])
        return orm(*a, **kw)
    cpm.RawModule = RawModule
    R = ART.RTSolver
    oa, ok, ost = R.alloc, R.k, R.step

    def alloc(s, need):
        r = oa(s, need)
        try:
            KK = max(int(s.K), int(getattr(s, "kcap", 0) or 0), int(getattr(s, "Kc", 0) or 0), 1)
            capN, capQ = int(s.capN), int(s.capQc) * 8
            nb = KK * capN * 4 * (3 * len(NODE3) + len(NODE1) + len(NODEI) + 6) + KK * capQ * 4 * (5 + 9) + capQ * 4 * 9
            s._ew = None
            torch.cuda.empty_cache()
            if nb > maxb:
                ST["on_alloc"] = f"끔 (버퍼 {nb / 1e9:.2f} GB > {maxb / 1e9:.1f})"
            else:
                dev = s.dev
                Z = lambda *sh: torch.zeros(*sh, device=dev, dtype=torch.float32)
                E = {nm: Z(KK, capN, 3) for nm in NODE3}
                E.update({nm: Z(KK, capN) for nm in NODE1})
                E.update({nm: torch.full((KK, capN), -7, device=dev, dtype=torch.int32) for nm in NODEI})
                E["F"] = torch.zeros(KK, capN, 3, device=dev, dtype=torch.float64)
                E["QS"] = Z(KK, capQ, 5)
                E["FQK"] = Z(KK, capQ, 9)
                E["FQ0"] = Z(capQ, 9)
                s._ew = E
                s._ew_KK = KK
                ST["on_alloc"] = f"켬 (버퍼 {nb / 1e9:.2f} GB, K {KK}, capN {capN}, capQ {capQ})"
            ST["nalloc"] += 1
        except Exception as e:
            s._ew = None
            ST["err"] = "alloc " + repr(e)
        _dump()
        return r

    def cp_(s, dst, src):
        if src is None:
            return
        n = min(dst.numel(), src.numel())
        s.dcopy(dst.view(-1)[:n], src.reshape(-1)[:n])

    def k(s, name, n, args, b=128):
        E = getattr(s, "_ew", None)
        if E is None:
            return ok(s, name, n, args, b)
        if name == "rt_reset":
            s._ew_k = -1
        kk = getattr(s, "_ew_k", -1)
        on = 0 <= kk < s._ew_KK
        if name in ENDK and on:
            for nm, src in (("Y", s.y), ("YH", s.yh), ("G0", getattr(s, "g0buf", None)), ("SK", s.sk), ("PM", s.pm), ("AN", getattr(s, "an", None)), ("ME", s.meff),
                            ("BG", getattr(s, "bGn", None)), ("BK", getattr(s, "bKn", None)), ("W8", getattr(s, "w8b", None)), ("KB", getattr(s, "kb", None)),
                            ("UL", getattr(s, "ulm", None)), ("F", s.f)):
                if src is not None and src.dtype == E[nm].dtype:
                    cp_(s, E[nm][kk], src)
        if name == "ao_commit" and on:
            ok(s, "esc_q", n, (args[0], args[1], args[2], args[3], args[4], args[5], P(E["QS"][kk])), 128)
            cp_(s, E["FQK"][kk], s.Fq)
        r = ok(s, name, n, args, b)
        if name in ("rt_ga_step0", "ao_step0_ga"):
            s._ew_k = kk = kk + 1
            if 0 <= kk < s._ew_KK:
                for nm, src in (("U0", s.U), ("V0", s.vk), ("A0", s.ak), ("Ut", s.Ut), ("Z0", s.z0)):
                    cp_(s, E[nm][kk], src)
        elif name in ENDK and on:
            for nm, src in (("U1", s.U), ("V1", s.vk)):
                cp_(s, E[nm][kk], src)
        elif name == "ao_commit" and on:
            cp_(s, E["A1"][kk], s.ak)
        elif name == "rt_gauss":
            cp_(s, E["FQ0"], s.Fq)
        return r

    def step(s, t):
        fr = int(round(t / s.h)) + 1
        win = ST["F0"] <= fr <= ST["F1"]
        pre = None
        if win:
            torch.cuda.synchronize()
            pre = {nm: getattr(s, nm).detach().clone() for nm in ("x", "v", "F", "pa", "perm")}
        r = ost(s, t)
        if win:
            try:
                save(s, fr, pre)
            except Exception as e:
                import traceback
                ST["err"] = f"save f{fr} " + traceback.format_exc()[-800:]
                _dump()
        return r
    R.alloc, R.k, R.step = alloc, k, step

    def save(s, fr, pre):
        torch.cuda.synchronize()
        E = getattr(s, "_ew", None)
        IP = s.IP.cpu().numpy().copy()
        o = dict(IP=IP, A=s.A.float().cpu().numpy(), Ai=s.Ai.float().cpu().numpy(), K=np.int32(getattr(s, "last_K", s.K)), h=np.float32(s.h), rec=np.int32(E is not None))
        perm = pre["perm"].long()

        def orig(t):
            t2 = torch.empty_like(t)
            t2[perm] = t
            return t2.cpu().numpy()
        for nm in ("x", "v", "F", "pa"):
            o["p0_" + nm] = orig(pre[nm])
            o["p1_" + nm] = orig(getattr(s, nm).detach())
        nn, nq = int(IP[9]), int(IP[12])
        if E is not None and nn > 0 and int(IP[13]) == 0:
            mn = s.mn[:nn]
            nid = torch.nonzero(mn > 0).squeeze(1)
            o["nid"] = nid.cpu().numpy().astype(np.int32)
            o["mn"] = mn[nid].cpu().numpy(); o["mqn"] = s.mqn[:nn][nid].cpu().numpy(); o["mv"] = s.mv.reshape(-1, 3)[:nn][nid].cpu().numpy()
            K = int(o["K"])
            for nm in NODE3:
                o[nm] = E[nm][:K, :nn][:, nid].cpu().numpy()
            for nm in NODE1 + NODEI:
                o[nm] = E[nm][:K, :nn][:, nid].cpu().numpy()
            o["Fn"] = E["F"][:K, :nn][:, nid].float().cpu().numpy()
            o["QS"] = E["QS"][:K, :nq].cpu().numpy(); o["FQK"] = E["FQK"][:K, :nq].cpu().numpy(); o["FQ0"] = E["FQ0"][:nq].cpu().numpy()
            o["Vq"] = s.Vq[:nq].cpu().numpy(); o["mq"] = s.mq[:nq].cpu().numpy(); o["xq0"] = s.xq0.reshape(-1, 3)[:nq].cpu().numpy()
            o["cb"] = s.cb.reshape(-1, 3)[:nq // 8].cpu().numpy()
            if getattr(s, "bx", None) is not None:
                o["bx"] = s.bx.reshape(-1, 4)[:nq].cpu().numpy()
            ST["nsave"] += 1
        else:
            ST["nskip"] += 1
        np.savez_compressed(os.path.join(ST["out"], "esc", f"f{fr:03d}.npz"), **o)
        _dump()
    import atexit
    atexit.register(_dump)
    print(f"[절 CM126] escwhy 켬 -- 프레임 {ST['F0']}~{ST['F1']} 걸음별 노드·가우스 점·입자 기록 -> {out}/esc (결과 불변)", flush=True)
    _dump()
