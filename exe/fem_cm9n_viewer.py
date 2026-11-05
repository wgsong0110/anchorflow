"""절 CM9.3: --nan_step 덤프 (nan_step_f<프레임>.npz) -> 표 (markdown) + 인터랙티브 뷰어 (three.js 단일 HTML).
사용: python exe/fem_cm9n_viewer.py <npz> <출력 html> <출력 표 md> [실행 이름]
정의 (뷰어·표에 그대로 적음):
  입자의 칸 = 프레임 시작 위치 기준 B-스플라인 칸 (base b = floor(L − ½), 노드 b..b+2 의 27 개, 그 칸의 2x2x2 가우스 점 8 개) -- 입자 위치는 프레임 끝 G2P 에서만 바뀌므로
  입자 NaN 시점 = 그 칸의 27 노드 변위 (y 또는 U) 또는 8 가우스 점 상태 (F_q, S_q) 가 처음 비유한이 된 걸음·뉴턴 반복 (없으면 'G2P 에서 처음').
  원인 = 칸에서 가장 먼저 생긴 것: (a) 노드 변위 비유한, (b) 가우스 점 시험 X 의 J ≤ 0 (유한), (c) 응력 (다시 계산) · 노드 기울기 비유한, (d) 가우스 점 상태 비유한; 같은 시점이면 a > b > c > d 순.
"""
import base64
import json
import sys

import numpy as np

UNSET = 0x7FFFFFFF


def decode(c, nt):
    c = int(c)
    if c == UNSET:
        return None
    sub = c & 1; c >>= 1; ph = c % 8; k = c // 8 - 1
    if k < 0:
        return "P2G (프레임 시작)"
    if ph == 0:
        return f"걸음 {k + 1} 시작점"
    if 1 <= ph <= nt:
        return f"걸음 {k + 1} 뉴턴 {ph} " + ("기울기" if sub == 0 else "끝")
    if ph == nt + 1:
        return f"걸음 {k + 1} 끝"
    return f"걸음 {k + 1} 커밋"


def main():
    npz, out_html, out_md = sys.argv[1:4]
    run = sys.argv[4] if len(sys.argv) > 4 else ""
    z = np.load(npz)
    nt = int(z["nt"][0]); K = int(z["K"][0]); fr = int(z["frame"][0]); cell = float(z["cell"][0])
    nan = z["nan"]; X = z["x_prev"]; N = len(nan); nn_ = int(nan.sum())
    cN, cQ, cJ = z["p_cN"].astype(np.int64), z["p_cQ"].astype(np.int64), z["p_cJ"].astype(np.int64)
    cS = np.minimum(z["p_cS"], z["p_cF"]).astype(np.int64)
    tN = np.minimum(cN, cQ)                                                            # 입자 NaN 시점 (칸 상태)
    # 원인: 가장 이른 것, 같으면 a > b > c > d
    st = np.stack([cN, cJ, cS, cQ], 1)
    first = st.min(1); cause = np.argmax(st == first[:, None], 1); cause[first == UNSET] = 4
    CAUSE = ["노드 변위 비유한 먼저", "칸 뒤집힘 (J ≤ 0) 먼저", "응력·기울기 비유한 먼저", "가우스 점 상태 비유한 먼저", "칸 안 이상 없음 (G2P 에서 처음)"]
    md = []
    md.append(f"# 수박 cm9 --gq_mass {run} 발산 프레임 {fr}: 걸음·뉴턴 반복별 NaN (절 CM9.3)\n")
    md.append(f"입자 {N:,} 중 프레임 {fr} 끝 NaN {nn_:,} (F 비유한 {int(z['Fnan'].sum()):,}). K {K} 걸음 x 뉴턴 {nt}. 칸 {cell * 1e3:.2f} mm. 전역 PCG 스칼라 비유한 처음: {decode(z['gsc'][0], nt) or '없음'}\n")
    # 1. NaN 시점 표
    codes = sorted(set(tN[nan].tolist()))
    md.append("## 1. NaN 입자의 NaN 시점 (칸 27 노드 변위·8 가우스 점 상태 중 처음 비유한)\n")
    md.append("| 시점 | NaN 입자 수 | 비율 % | 그 시점 처음 비유한이 된 노드 수 (변위) |\n|---|---|---|---|")
    ncnt = {}
    for c in np.unique(z["n_mN"]):
        ncnt[int(c)] = int((z["n_mN"] == c).sum())
    groups = []
    for c in codes:
        n = int(((tN == c) & nan).sum())
        lab = decode(c, nt) or "G2P 에서 처음 (칸 상태 유한)"
        md.append(f"| {lab} | {n:,} | {100 * n / max(nn_, 1):.2f} | {ncnt.get(int(c), 0) if c != UNSET else '-'} |")
        groups.append(dict(code=int(c), label=lab, n=n))
    nonnan_marked = int(((tN != UNSET) & ~nan).sum())
    md.append(f"\nNaN 이 아닌 입자 중 칸 상태가 비유한이 된 입자: {nonnan_marked:,} (G2P 가 그 노드 몫을 0 가중으로 쓴 경우 등)\n")
    # 2. 원인 표
    md.append("## 2. 원인 분류 (칸에서 가장 먼저 생긴 것; 같은 시점이면 a > b > c > d)\n")
    md.append("| 원인 | NaN 입자 수 | 비율 % | 그 원인의 시점 분포 (상위 3) |\n|---|---|---|---|")
    for i, nm in enumerate(CAUSE):
        m = nan & (cause == i)
        n = int(m.sum())
        if i < 4 and n:
            u, cnt = np.unique(first[m], return_counts=True); o = np.argsort(-cnt)[:3]
            top = ", ".join(f"{decode(u[j], nt)} {cnt[j]:,}" for j in o)
        else:
            top = "-"
        md.append(f"| {nm} | {n:,} | {100 * n / max(nn_, 1):.2f} | {top} |")
    # 3. 시점별 처음 생긴 것 (전체 몸, 노드·점 수)
    md.append("\n## 3. 시점별 처음 생긴 노드·가우스 점 수 (몸 전체)\n")
    md.append("| 시점 | 노드 변위 비유한 | 노드 기울기 비유한 | 노드 속도·가속도 비유한 | 점 J ≤ 0 (유한) | 점 응력 비유한 | 점 상태 비유한 |\n|---|---|---|---|---|---|---|")
    allc = sorted(set(np.concatenate([z[k] for k in ("n_mN", "n_mF", "n_mV", "q_mJ", "q_mS", "q_mQ")]).tolist()) - {UNSET})
    for c in allc:
        md.append(f"| {decode(c, nt)} | " + " | ".join(str(int((z[k] == c).sum())) for k in ("n_mN", "n_mF", "n_mV", "q_mJ", "q_mS", "q_mQ")) + " |")
    # 4. 큰 증분 노드
    jm, jk, mr = z["n_jm"], z["n_jk"], z["n_mrel"]
    lo, nb, A = z["n_lo"], z["n_nb"], z["A"]
    md.append(f"\n## 4. 한 걸음 변위 증분이 큰 노드 (|U_k − U_(k−1)|, 칸 단위; 비유한 값은 빼고 유한한 걸음 중 최대)\n")
    md.append(f"질량 있는 노드 증분 중앙값 {np.median(jm[mr > 0]):.3f} 칸, 99.9 % {np.percentile(jm[mr > 0], 99.9):.3f} 칸; 1 칸 넘는 노드 {int((jm > 1).sum())}, 5 칸 넘는 노드 {int((jm > 5).sum())} (거의 빈 노드 문턱 = 질량 중앙값의 {float(z['mthr_rel'][0]):.0e} 배)\n")
    md.append("| 노드 | 증분 (칸) | 걸음 | 질량 / 중앙값 | 변위 비유한 시점 |\n|---|---|---|---|---|")
    top = np.argsort(-jm)[:15]
    for i in top:
        md.append(f"| {i} | {jm[i]:.2f} | {int(jk[i]) + 1} | {mr[i]:.2e} | {decode(z['n_mN'][i], nt) or '유한'} |")
    pj = z["p_jmax"]
    md.append(f"\nNaN 입자 칸의 27 노드 최대 증분: 중앙값 {np.median(pj[nan]):.2f} 칸, 1 칸 넘는 입자 {int((pj[nan] > 1).sum()):,} / {nn_:,}; NaN 아닌 입자: 중앙값 {np.median(pj[~nan]):.3f} 칸\n")
    open(out_md, "w").write("\n".join(md) + "\n")

    # ---- 뷰어
    rng = np.random.default_rng(0)
    gi = np.where(~nan & np.isfinite(X).all(1))[0]
    gs = np.sort(rng.choice(gi, min(200000, len(gi)), replace=False))
    ni = np.where(nan)[0]
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).astype(np.float32).tobytes()).decode()
    # 시점 그룹 (뷰어 색 = 시점 순서 색표)
    G = []
    for g in groups:
        idx = ni[tN[ni] == g["code"]]
        G.append(dict(label=g["label"], n=len(idx), pos=b64(X[idx])))
    C = []
    for i, nm in enumerate(CAUSE):
        idx = ni[cause[ni] == i]
        C.append(dict(label=nm, n=len(idx), pos=b64(X[idx])))
    # 큰 증분 노드 위치 (프레임 시작 격자 = 이 프레임 노드 위치), 1 칸 넘는 것
    jj = np.where(jm > 1.0)[0]
    nb1, nb2 = int(nb[1]), int(nb[2])
    Lj = np.stack([jj // (nb1 * nb2), (jj // nb2) % nb1, jj % nb2], 1) + lo[None, :]
    PJ = (Lj.astype(np.float64) @ A.T.astype(np.float64))
    # J ≤ 0 가 처음 생긴 가우스 점 (위치 = 칸 base + 1 ± 1/(2√3) 격자 좌표)
    qj = np.where(z["q_mJ"] != UNSET)[0]
    cb = z["c_cb"].astype(np.float64); h = 0.5 / np.sqrt(3.0)
    oc = qj & 7; off = np.stack([(oc >> 2) & 1, (oc >> 1) & 1, oc & 1], 1) * 2 * h - h
    Lq = cb[qj >> 3] + lo[None, :] + 1.0 + off
    PQ = Lq @ A.T.astype(np.float64)
    meta = dict(frame=fr, run=run, N=N, n_nan=nn_, n_gray=len(gs), n_rest=int(len(gi)), K=K, nt=nt,
                groups=[dict(label=g["label"], n=g["n"]) for g in G], causes=[dict(label=c["label"], n=c["n"]) for c in C],
                jump=dict(n=len(jj), jm=[round(float(q), 2) for q in jm[jj]], st=[int(q) + 1 for q in jk[jj]], mr=[float(f"{q:.3g}") for q in mr[jj]],
                          nf=[decode(q, nt) or "유한" for q in z["n_mN"][jj]]),
                jq=dict(n=len(qj), lab=[decode(q, nt) for q in z["q_mJ"][qj]]),
                lo=np.nanmin(X[gs], 0).tolist(), hi=np.nanmax(X[gs], 0).tolist())
    tpl = open(__file__.replace("fem_cm9n_viewer.py", "fem_cm9n_viewer_tpl.html")).read()
    html = (tpl.replace("__META__", json.dumps(meta, ensure_ascii=False)).replace("__GRAY__", b64(X[gs]))
            .replace("__GROUPS__", json.dumps([g["pos"] for g in G])).replace("__CAUSES__", json.dumps([c["pos"] for c in C]))
            .replace("__PJ__", b64(PJ)).replace("__PQ__", b64(PQ)))
    open(out_html, "w").write(html)
    print(open(out_md).read())
    print(f"[뷰어] {out_html} {len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
