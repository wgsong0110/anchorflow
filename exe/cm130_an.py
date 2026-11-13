"""절 CM130 (로컬 numpy, 진단): escwhy 기록 (esc/fNNN.npz, K 1) 으로 같은 걸음의 가우스 점 psi 와 입자 psi 를 대조.
- 가우스 점 끝 psi (뉴턴이 보는 값): 마지막 커밋 앞 시험 X 의 특이값 QS[:, :3]·DP 상태 QS[:, 3] 에 정답 PG 반환 사상 (ao_dpf 와 같은 식,
  eps = log|sigma|; 0 탄성 그대로 · 1 꼬리 tr/3 + s eps^ · 2 꼭짓점 0) 을 건 탄성 eps 로 Hencky psi. 뒤집힘 = det X (QS[:, 4]) <= 0.
- 가우스 점 시작 psi: FQ0 (프레임 시작 rt_gauss = 입자 -> 가우스 점) 의 SVD.
- 입자 psi: p0_F (프레임 시작), p1_F (프레임 끝 = 가우스 점 보간 교체 뒤). 부피 = 입자 질량/밀도 (모두 같다고 보고 총 부피/N), 가우스 점 = Vq.
- 입자 -> 칸: 칸 중심 = cb 의 선형 사상 (xq0 8 점 평균에 맞춤), 입자 프레임 시작 x 의 격자 좌표 반올림. 칸 안 삼선형 가중 (fem_fpi 의 t·w 식) 으로
  입자 자리의 가우스 점 끝 psi 가중 평균·최대, 둘레 뒤집힌 점 여부 (보간이 유클리드 평균으로 가는 조건).
- 자리: 바닥 근처 (z − z_floor < 0.03 m), 확장 칸 (27 노드 중 W8 >= 0 인 외부 노드가 있음), 꼭짓점 점 (상태 2), 방출 앞면 (27 노드 중 1e-3 < kb <= 0.999).
사용: python exe/cm130_an.py <판 디렉토리 (esc/ 있음)> <이름> [z_floor 0.5]"""
import glob, json, os, sys
import numpy as np
MU, LAM, CDP = 1.9230769e7, 2.8846154e7, 1.0614514
run, nm = sys.argv[1:3]; ZF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
G = 0.28867513459481287


def hencky(e):
    return MU * (e ** 2).sum(-1) + 0.5 * LAM * e.sum(-1) ** 2


def psi_F(F):
    s = np.linalg.svd(F.reshape(-1, 3, 3).astype(np.float64), compute_uv=False)
    return hencky(np.log(np.maximum(np.abs(s), 1e-14))), np.linalg.det(F.reshape(-1, 3, 3).astype(np.float64))


def psi_rm(S, st):
    e = np.log(np.maximum(np.abs(S.astype(np.float64)), 1e-14)); tr = e.sum(1, keepdims=True); eh = e - tr / 3
    rho = np.linalg.norm(eh, axis=1, keepdims=True)
    s = np.where(rho > 0, -CDP * tr / np.maximum(rho, 1e-30), 0.0)
    eE = np.where(st[:, None] == 1, tr / 3 + s * eh, e)
    eE = np.where(st[:, None] == 2, 0.0, eE)
    return hencky(eE), hencky(e)


def wq(s, v, P):
    return f"{(v * s).sum():.4g}"


out = {}; rows = []
fs = sorted(glob.glob(os.path.join(run, "esc", "f*.npz")))
print(f"[{nm}] {run} 프레임 {len(fs)}")
hdr = ("f | 가우스점 E 끝 (RM 뒤 = 뉴턴) · RM 앞 시험 · 시작 FQ0 [J] | 입자 E 시작 · 끝 [J] | 끝 입자/가우스 | 뒤집힌 점 수 · 그 psi 몫 | 뒤집힌 입자 수 (둘레 뒤집힌 점 있음) · psi 몫 | "
       "입자 psi>1e6: 수 · 그 자리 가우스 psi 가중평균 중앙 · 입자/자리 중앙")
print(hdr)
for f_ in fs:
    z = np.load(f_); fr = int(os.path.basename(f_)[1:4])
    if int(z["rec"]) == 0 or "QS" not in z.files:
        print(f"{fr} 기록 없음"); continue
    IP = z["IP"]; nq = int(IP[12]); K = int(z["K"])
    QS = z["QS"][K - 1]; Vq = z["Vq"].astype(np.float64); xq = z["xq0"].astype(np.float64); cb = z["cb"].astype(np.int64)
    st = QS[:, 3].astype(int); dX = QS[:, 4]
    pq, pqX = psi_rm(QS[:, :3], st)
    p0q, _ = psi_F(z["FQ0"])
    N = z["p0_F"].shape[0]; vp = Vq.sum() / N  # 입자 부피 근사 (질량 같음 가정)
    pp0, d0 = psi_F(z["p0_F"]); pp1, d1 = psi_F(z["p1_F"])
    # 입자 -> 칸
    nc = len(cb); xc = xq.reshape(nc, 8, 3).mean(1)
    M = np.c_[cb, np.ones(nc)]; co = np.linalg.lstsq(M, xc, rcond=None)[0]; Am = co[:3].T; o = co[3]
    lat = np.linalg.solve(Am, (z["p0_x"].astype(np.float64) - o).T).T; pb = np.rint(lat).astype(np.int64); l = lat - pb
    key = lambda a: (a[:, 0] * 200003 + a[:, 1]) * 200003 + a[:, 2]
    kc = key(cb); order = np.argsort(kc); ks = kc[order]
    kp = key(pb); ii = np.searchsorted(ks, kp); ii = np.clip(ii, 0, nc - 1); pc = np.where(ks[ii] == kp, order[ii], -1)
    t = np.clip((l + G) / (2 * G), 0, 1)
    W = np.zeros((N, 8)); I = np.zeros((N, 8), np.int64)
    for oc in range(8):
        W[:, oc] = (t[:, 0] if oc & 4 else 1 - t[:, 0]) * (t[:, 1] if oc & 2 else 1 - t[:, 1]) * (t[:, 2] if oc & 1 else 1 - t[:, 2])
        I[:, oc] = np.maximum(pc, 0) * 8 + oc
    ok = pc >= 0
    W[~ok] = 0
    loc = (W * pq[I]).sum(1); locmax = np.where(W > 0, pq[I], 0).max(1)
    inv = dX <= 0; pinv = (W > 0) & inv[I]; anyb = pinv.any(1)
    # 자리 분류 (가우스 점)
    nid = z["nid"].astype(np.int64); nb1, nb2 = int(IP[7]), int(IP[8])
    W8 = z["W8"][K - 1]; KB = z["KB"][K - 1]
    nn = int(IP[9]); w8 = np.full(nn + 1, -1, np.int64); kb = np.full(nn + 1, 1.0); w8[nid] = W8; kb[nid] = KB
    ext = np.zeros(nc, bool); front = np.zeros(nc, bool)
    for o_ in range(27):
        i, j, k = o_ // 9, (o_ // 3) % 3, o_ % 3
        ids = ((cb[:, 0] + i) * nb1 + (cb[:, 1] + j)) * nb2 + (cb[:, 2] + k)
        ids = np.where((ids >= 0) & (ids < nn), ids, nn)
        ext |= w8[ids] >= 0; front |= (kb[ids] > 1e-3) & (kb[ids] <= 0.999)
    extq = np.repeat(ext, 8); frq = np.repeat(front, 8)
    flq = (xq[:, 2] - ZF) < 0.03; apq = st == 2
    Eq = (Vq * pq).sum(); EqX = (Vq * pqX).sum(); Eq0 = (Vq * p0q).sum(); Ep0 = vp * pp0.sum(); Ep1 = vp * pp1.sum()
    ip1 = d1 <= 0
    big = pp1 > 1e6
    r_ = dict(f=fr, nq=nq, N=N, Eq=Eq, EqX=EqX, Eq0=Eq0, Ep0=Ep0, Ep1=Ep1, nqinv=int((inv & (Vq > 0)).sum()), Eq_inv=float((Vq * pq)[inv].sum()),
              npinv=int(ip1.sum()), npinv_anyb=int((ip1 & anyb).sum()), Ep_inv=float(vp * pp1[ip1].sum()), nbig=int(big.sum()),
              nanyb=int(anyb.sum()), Ep_anyb=float(vp * pp1[anyb].sum()), Eloc_anyb=float(vp * loc[anyb].sum()), Eloc=float(vp * loc.sum()),
              unmapped=int((~ok).sum()))
    # 큰 에너지 가우스 점 (상위 1 % psi 몫) 의 자리 몫
    Eqv = Vq * pq; srt = np.argsort(Eqv)[::-1]; top = srt[:max(1, nq // 100)]
    for nmz, msk in (("floor", flq), ("ext", extq), ("apex", apq), ("front", frq), ("inv", inv)):
        r_[f"Eq_{nmz}"] = float(Eqv[msk].sum() / max(Eq, 1e-30)); r_[f"nq_{nmz}"] = int((msk & (Vq > 0)).sum())
        r_[f"inv_{nmz}"] = int((inv & msk & (Vq > 0)).sum())
    r_["Eq_top1pct"] = float(Eqv[top].sum() / max(Eq, 1e-30))
    r_["ratio_p_over_loc_big_med"] = float(np.median(pp1[big] / np.maximum(loc[big], 1e-9))) if big.any() else None
    r_["loc_big_med"] = float(np.median(loc[big])) if big.any() else None
    # 뒤집힌 점: 상태별, 높이
    if inv.any():
        r_["inv_state"] = [int(((st == s_) & inv).sum()) for s_ in (0, 1, 2)]
        r_["inv_zmed"] = float(np.median(xq[inv, 2] - ZF)); r_["inv_psi_med"] = float(np.median(pq[inv]))
        r_["inv_detmin"] = float(dX[inv].min())
    if ip1.any():
        r_["pinv_zmed"] = float(np.median(z["p1_x"][ip1, 2] - ZF)); r_["pinv_psi_med"] = float(np.median(pp1[ip1]))
        r_["pinv_loc_med"] = float(np.median(loc[ip1])); r_["pinv_wfrac_med"] = float(np.median((W * inv[I]).sum(1)[ip1]))
    # 보간 영향 (뒤집힌 점 없는 둘레의 입자): 입자 psi 대 자리 가중 psi
    nb = ok & ~anyb
    r_["E_noinv_part"] = float(vp * pp1[nb].sum()); r_["E_noinv_loc"] = float(vp * loc[nb].sum())
    rows.append(r_)
    print(f"{fr:2d} | {Eq:.3g} · {EqX:.3g} · {Eq0:.3g} | {Ep0:.3g} · {Ep1:.3g} | {Ep1 / max(Eq, 1e-30):.3g} | {r_['nqinv']} · {r_['Eq_inv'] / max(Eq, 1e-30):.3f} | "
          f"{r_['npinv']} ({r_['npinv_anyb']}) · {r_['Ep_inv'] / max(Ep1, 1e-30):.3f} | {r_['nbig']} · {r_['loc_big_med']} · {r_['ratio_p_over_loc_big_med']}", flush=True)
    print(f"   자리 몫 (가우스 E): 바닥 {r_['Eq_floor']:.3f} ({r_['nq_floor']} 점) 확장 {r_['Eq_ext']:.3f} ({r_['nq_ext']}) 꼭짓점 {r_['Eq_apex']:.3f} ({r_['nq_apex']}) 앞면 {r_['Eq_front']:.3f} ({r_['nq_front']}) "
          f"뒤집힘 {r_['Eq_inv']/max(Eq,1e-30):.3f} 상위1% {r_['Eq_top1pct']:.3f} | 뒤집힌 점 자리: 바닥 {r_['inv_floor']} 확장 {r_['inv_ext']} 꼭짓점 {r_['inv_apex']} 앞면 {r_['inv_front']} "
          f"상태 {r_.get('inv_state')} z중앙 {r_.get('inv_zmed')} psi중앙 {r_.get('inv_psi_med')} det최소 {r_.get('inv_detmin')}", flush=True)
    print(f"   입자 둘레 뒤집힌 점 있음 {r_['nanyb']} 개: 입자 E {r_['Ep_anyb']:.3g} 대 자리 가중 가우스 E {r_['Eloc_anyb']:.3g} | 둘레 뒤집힘 없음: 입자 E {r_['E_noinv_part']:.3g} 대 자리 {r_['E_noinv_loc']:.3g} | "
          f"뒤집힌 입자 z중앙 {r_.get('pinv_zmed')} psi중앙 {r_.get('pinv_psi_med')} 자리psi중앙 {r_.get('pinv_loc_med')} 둘레 뒤집힌 가중 중앙 {r_.get('pinv_wfrac_med')} | 칸 못 찾은 입자 {r_['unmapped']}", flush=True)
json.dump(rows, open(os.path.join(run, f"cm130_an_{nm}.json"), "w"))
print("CM130_AN_DONE")
