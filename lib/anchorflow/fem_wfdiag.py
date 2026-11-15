"""wolffix 진단 (11-15, 결과 불변): wolf 마지막 무너짐·튐 프레임의 뉴턴 반복별 노드 잔차 분해.
그래프 안 복사만 (커널 순서·인자 그대로). 반복 (걸음 k, 반복 it) 마다:
  - rt_bar_n 직전 f (장벽 전 힘: 탄성·고스트·구속), rt_bar_n 직후 f (장벽 포함), rt_grad 뒤 g0buf (= −g, g = Mi η + T f), y,
  - rt_kstep 뒤 kk (장벽 실현 가능 보폭 2^-k).
걸음 0 반복 0 의 rt_meff 뒤: meff, mn, sk, pm, U, Ut.
창 (env WFD_F0..WFD_F1, 1 부터) 프레임마다 질량 > 0 노드만 <out>/wfd/f%03d.npz 로 (분석은 exe/wfd_an.py)."""
import json
import os

import numpy as np
import torch

ST = dict(out=".", F0=1, F1=0, nalloc=0, nframe=0, nwrite=0, err=None)
KMAX = 4


def install(out):
    from . import fem_an_rt as ART
    ST.update(out=out, F0=int(os.environ.get("WFD_F0", "1")), F1=int(os.environ.get("WFD_F1", "0")))
    os.makedirs(os.path.join(out, "wfd"), exist_ok=True)
    R = ART.RTSolver
    oa, ok, octl = R.alloc, R.k, R.ctl

    def alloc(s, need):
        r = oa(s, need)
        try:
            capN = s.mn.numel()
            nt = int(s.nt); NR = KMAX * nt
            dev = s.dev
            s.wfF0 = torch.zeros(NR, capN * 3, device=dev, dtype=torch.float64)
            s.wfF1 = torch.zeros(NR, capN * 3, device=dev, dtype=torch.float64)
            s.wfG = torch.zeros(NR, capN * 3, device=dev, dtype=torch.float32)
            s.wfY = torch.zeros(NR, capN * 3, device=dev, dtype=torch.float32)
            s.wfK = torch.zeros(NR, 1, device=dev, dtype=torch.int32)
            s.wfDG = torch.zeros(NR, s.SC.numel(), device=dev, dtype=s.SC.dtype)
            s.wfN = {nm: torch.zeros_like(getattr(s, nm)) for nm in ("meff", "mn", "sk", "pm", "U", "Ut", "eta")}
            s.wfIP = torch.zeros_like(s.IP)
            s._wf_k, s._wf_it = -1, 0
            ST["nalloc"] += 1
        except Exception as e:
            ST["err"] = "alloc " + repr(e)
        return r

    def row(s):
        k = max(s._wf_k, 0)
        return k * int(s.nt) + s._wf_it if k < KMAX else None

    def ctl(s, op, it=0):
        r = octl(s, op, it)
        if getattr(s, "wfG", None) is not None:
            if op == 0:
                s._wf_k = -1
            elif op == 1:
                s._wf_k += 1
            elif op == 2:
                s._wf_it = it
        return r

    def k(s, name, n, args, b=128):
        if getattr(s, "wfG", None) is not None and name == "rt_bar_n":
            i = row(s)
            if i is not None:
                s.dcopy(s.wfF0[i, :s.f.numel()], s.f)
        r = ok(s, name, n, args, b)
        if getattr(s, "wfG", None) is not None:
            i = row(s)
            if i is not None:
                if name == "rt_bar_n":
                    s.dcopy(s.wfF1[i, :s.f.numel()], s.f)
                elif name == "rt_grad":
                    s.dcopy(s.wfG[i, :s.g0buf.numel()], s.g0buf); s.dcopy(s.wfY[i, :s.y.numel()], s.y)
                elif name == "rt_kstep":
                    s.dcopy(s.wfK[i], s.kk[:1])
                elif name == "rt_lsup":
                    s.dcopy(s.wfDG[i], s.SC)
                elif name == "rt_meff" and s._wf_k == 0:
                    for nm, t in s.wfN.items():
                        s.dcopy(t, getattr(s, nm))
                    s.dcopy(s.wfIP, s.IP)
        return r
    R.alloc, R.k, R.ctl = alloc, k, ctl

    def frame(fn, s, fi, *a):
        if getattr(s, "_wf_in", False):
            return fn(s, *a)
        s._wf_in = True
        try:
            f_ = fi + 1
            if getattr(s, "wfG", None) is not None:
                s.wfK.fill_(-1); s.wfG.zero_()
            r = fn(s, *a)
            if ST["F0"] <= f_ <= ST["F1"] and getattr(s, "wfG", None) is not None:
                try:
                    torch.cuda.synchronize()
                    IP = s.wfIP.cpu().numpy(); nn = int(IP[9])
                    mn = s.wfN["mn"][:nn].cpu().numpy()
                    idx = np.nonzero((mn > 0) | (s.wfN["pm"][:nn].cpu().numpy() != 0))[0]
                    nt = int(s.nt); K = int(getattr(s, "Kcur", 1))
                    i3 = (idx[:, None] * 3 + np.arange(3)[None]).reshape(-1)
                    nb1, nb2 = int(IP[7]), int(IP[8])
                    L = np.stack([idx // (nb1 * nb2) + IP[0], (idx // nb2) % nb1 + IP[1], idx % nb2 + IP[2]], 1).astype(np.float64)
                    A = s.A.double().cpu().numpy().reshape(3, 3)
                    rows = min(K, KMAX) * nt
                    D = dict(idx=idx, L=L, A=A, IP=IP, K=K, nt=nt,
                             meff=s.wfN["meff"][:nn].cpu().numpy()[idx], mn=mn[idx], sk=s.wfN["sk"][:nn].cpu().numpy()[idx], pm=s.wfN["pm"][:nn].cpu().numpy()[idx],
                             U=s.wfN["U"][:nn * 3].cpu().numpy().reshape(-1, 3)[idx], Ut=s.wfN["Ut"][:nn * 3].cpu().numpy().reshape(-1, 3)[idx],
                             eta0=s.wfN["eta"][:nn * 3].cpu().numpy().reshape(-1, 3)[idx],
                             F0=s.wfF0[:rows, :].cpu().numpy()[:, i3].reshape(rows, -1, 3).astype(np.float32),
                             F1=s.wfF1[:rows, :].cpu().numpy()[:, i3].reshape(rows, -1, 3).astype(np.float32),
                             G=s.wfG[:rows, :].cpu().numpy()[:, i3].reshape(rows, -1, 3),
                             Y=s.wfY[:rows, :].cpu().numpy()[:, i3].reshape(rows, -1, 3),
                             KK=s.wfK[:rows, 0].cpu().numpy(), SC=s.wfDG[:rows].cpu().numpy(), DG=s.DG[:2 * rows].cpu().numpy())
                    np.savez_compressed(os.path.join(ST["out"], "wfd", "f%03d.npz" % f_), **D)
                    ST["nwrite"] += 1
                except Exception as e:
                    ST["err"] = "read " + repr(e)[:300]
            ST["nframe"] += 1
            with open(os.path.join(ST["out"], "wfdinfo.json"), "w") as f2:
                json.dump(ST, f2)
            return r
        finally:
            s._wf_in = False

    osplit = ART.split_step

    def split_step(s, t, h, depth):
        if depth > 0:
            return osplit(s, t, h, depth)
        return frame(lambda s_, *a: osplit(s_, *a), s, int(round(t / h)), t, h, depth)
    ART.split_step = split_step
    try:
        from . import fem_wmrx as WRX
        ocol = WRX.split_col

        def split_col(s, t, h, sub_dt, depth, fn=None):
            if depth > 0:
                return ocol(s, t, h, sub_dt, depth, fn)
            return frame(lambda s_, *a: ocol(s_, *a), s, int(round(t / h)), t, h, sub_dt, depth, fn)
        WRX.split_col = split_col
    except Exception as e:
        ST["err"] = "wrx " + repr(e)
    print(f"[wfdiag] 프레임 {ST['F0']}~{ST['F1']} 노드 잔차 분해 기록 (결과 불변)", flush=True)
