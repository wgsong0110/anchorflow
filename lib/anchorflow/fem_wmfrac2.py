"""절 CM32 (조정자 결정 11-08, 정의 변경): 수박 첫 접촉 -- 절 CM30 (a)+(b) 위에 절 CM29 (a) 뒤집힘 필터 노드별 국소화를 얹은 판 + 진단 (시도별 기록).
옛 재질 실행기 (fem_ao_rt.AORTSolver) 의 절 CM30 (b) 판 (--ws_bar R) 에만, s.wf["ws"] 에 fi (국소 필터) 또는 rec (기록) 가 있을 때만 이 newton 이
fem_wmsparse.newton 을 대신함 -- 없으면 실행·소스 글자 그대로 (새 CUDA 커널 없음: 절 CM30 커널 ws_* 와 torch 복사만).

fi   절 CM30 (b) 는 노드 보폭 2^-kn 을 모두 원래 전역 뒤집힘 필터 kg 로 시작해 확인 라운드에서 관통할 표본점·뒤집힘 판정을 어기는 가우스 점의
     27 노드에만 kn + k_q 를 더함. 여기서는 kn = 0 (온 보폭) 으로 시작 -- 첫 라운드 (온 걸음 후보) 에서 필터가 거는 가우스 점 (k_q > 0) 의 27 노드에
     k_q (= 절 CM29 (a) 첫 판정 kn = 걸친 점 k_q 최대) 와 관통할 표본점의 27 노드에 k_q, 그 뒤 R 라운드는 절 CM30 (b) 와 같음 (라운드 R + 1).
     최종 확인에서 어기면 그 반복은 원래 전역 보폭 2^-max(kg, kb) 로 되돌림 (실현 가능·뒤집힘 없음 보장). SC[14] = 노드 보폭 최소 (덜 풀린 걸음 판정).
rec  진단 (결과 불변): 시도 (K = 1, 2, 4, 8) 마다 걸음·뉴턴 반복별 [돌았나, 전역 필터 kg, 전역 장벽 kb (온 걸음 후보), 최종 노드 k 최대,
     라운드별 (관통 표본점 수, 뒤집힘 가우스 점 수) ...] 와 SC[14]. 프레임마다 그 프레임의 시도만 읽음 (split_step tries).
"""
import numpy as np
import torch

from .fem_ao_rt import P

W = 24                                                                                # 기록 한 줄 int 수
NROW = 64                                                                             # 걸음 x 반복 칸 (K 8 x 뉴턴 8)
NATT = 8


def on(s):
    w = getattr(s, "wf", {}).get("ws", {})
    return int(w.get("bar", 0)) > 0 and (bool(w.get("fi")) or bool(w.get("rec")))


def rec_alloc(s):
    """실행기에서 풀이기 만든 뒤 한 번 (용량과 무관한 크기)."""
    s.w2R = torch.zeros(NATT, NROW, W, device=s.dev, dtype=torch.int32)
    s.w2S = torch.zeros(NATT, NROW, device=s.dev, dtype=torch.float64)


def newton(s, kstep, it, capN, capQ, nr, thv, WB, NR):
    """fem_wmsparse.newton 과 같은 순서·커널; fi 면 노드 k 시작 0 + 라운드 하나 더, rec 면 반복마다 기록 (torch 복사)."""
    i32, f = np.int32, np.float32
    IP = s.IP; ws = s.wf["ws"]; R = int(ws["bar"]); fi = bool(ws.get("fi")); F3 = [f(q) for q in nr]
    RR = R + 1 if fi else R
    assert WB is not None and s.lsinv and 2 * (RR + 1) <= 64
    rec = getattr(s, "w2R", None) is not None
    a_ = int(s.K).bit_length() - 1
    j = kstep * s.nt + it
    if kstep == 0 and it == 0:
        s.k("ws_zero", 8, (i32(8), P(s.wsst)), b=32)
        if rec:
            s.w2R[a_].fill_(-1); s.w2S[a_].fill_(-1.0)
    if rec:
        s.wskg.fill_(-1)                                                             # 허용치로 멈춘 반복은 ws_init 이 안 돌아 -1 로 남음
    s.k("rt_ytry", capN, (P(IP), P(s.sk), P(s.pm), *F3, P(s.yh), P(s.eta), P(s.dl), P(s.wbyc)), b=256)
    s.zf(s.kk)
    s.k("rt_bar_kfe", capQ, (P(IP), P(s.cb), P(s.BW), P(s.y), P(s.wbyc), P(s.U), thv, *F3, f(WB["PN"]), P(s.kk)))
    s.k("ws_init", capN, (P(IP), P(s.SC), P(s.wskn), P(s.wskn2), P(s.wskg)), b=256)
    if fi:
        s.zf(s.wskn)                                                                 # 절 CM32: 노드 k 시작 = 0 (온 보폭) -- 필터도 국소로
    s.zf(s.wsfl)
    aend = s._aw_end()
    bargs = lambda r_, sc: (P(IP), P(s.cb), P(s.BW), P(s.y), P(s.wsy1), P(s.U), thv, *F3, f(WB["PN"]), P(s.wskn), P(s.wskn2), P(s.wsfl[2 * r_:]), i32(sc))

    def check(r_, sc):
        s.k("ws_cand", capN, (P(IP), P(s.wskn), P(s.sk), P(s.pm), *F3, P(s.yh), P(s.eta), P(s.dl), P(s.awtd), P(s.wsy1), P(s.wsdd)), b=256)
        s.zf(s.wskq); s.zf(s.wskk)
        s.k("ws_inv_q", capQ, (P(IP), P(s.cb), P(s.BG), P(s.Fq), P(s.Vq), P(s.y), P(s.wsdd), P(s.wskk)) + aend + (P(s.wskq),))
        s.k("ws_nk", capQ, (P(IP), P(s.cb), P(s.wskq), P(s.wskn), P(s.wskn2), P(s.wsfl[2 * r_:]), i32(sc)))
        s.k("ws_bk", capQ, bargs(r_, sc))
    for r_ in range(RR):
        check(r_, 1)
        s.k("ws_merge", capN, (P(IP), P(s.wskn), P(s.wskn2)), b=256)
    check(RR, 0)
    s.zf(s.wskm)
    s.k("ws_fb", capN, (P(IP), P(s.wsfl[2 * RR:]), P(s.wskg), P(s.kk), P(s.wskn), P(s.wskm), P(s.wsst)), b=256)
    s.k("ws_sc", 1, (P(s.SC), P(s.wskm), P(s.wskg), P(s.wsst)), b=1)
    if NR:
        s.NRkb[j:j + 1].copy_(s.kk[:1]); s.NRst[j:j + 1].copy_(s.SC[14:15])
    if rec and j < NROW:
        row = s.w2R[a_, j]
        row[1:2].copy_(s.wskg[:1]); row[2:3].copy_(s.kk[:1]); row[3:4].copy_(s.wskm[:1])
        row[4:4 + 2 * (RR + 1)].copy_(s.wsfl[:2 * (RR + 1)])
        s.w2S[a_, j:j + 1].copy_(s.SC[14:15])
    s.k("ws_lsup", capN, (P(IP), P(s.wskn), P(s.sk), P(s.pm), *F3, P(s.yh), P(s.eta), P(s.dl), P(s.y), i32(1 if s.compact else 0)), b=256)


def frame_rec(s, tries):
    """프레임 시도별 기록 (호스트). tries = split_step 의 [(K, 덜 풀린 걸음 수, ms)] (쪼개기 없으면 [(K, -, -)])."""
    ws = s.wf["ws"]; RR = int(ws["bar"]) + (1 if ws.get("fi") else 0)
    R_ = s.w2R.cpu().numpy(); S_ = s.w2S.cpu().numpy()
    out = []
    for t in tries:
        K = int(t[0]); a_ = K.bit_length() - 1
        nt = int(s.wf.get("ks_nt") or s.nt) if K > 1 else int(s.nt)
        its = []
        for j in range(min(K * nt, NROW)):
            r = R_[a_, j]
            if r[1] < 0:                                                             # 허용치로 멈춘 반복 (또는 이 시도에 없음)
                continue
            fl = r[4:4 + 2 * (RR + 1)].reshape(-1, 2)
            its.append(dict(k=j // nt, i=j % nt, kg=int(r[1]), kb=int(r[2]), km=int(r[3]), fb=int(fl[-1].sum() > 0),
                            bar=[int(q) for q in fl[:, 0]], inv=[int(q) for q in fl[:, 1]], st=float(S_[a_, j])))
        out.append(dict(K=K, nf=(int(t[1]) if len(t) > 1 and t[1] is not None else None), it=its))
    return out
