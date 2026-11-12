"""절 CM117 집계 (CPU, 읽기만): wolf 수렴·확장 가르기 판 (a9 H9 · b9 H9 수렴 · c9 H9 확장 끔 · d5 H5) -> <O>/convabl_an.json.
사용: python exe/convabl_an.py <O> <판> ...
 - 오차·발산·반복·프레임 시간: summary.json. 높이: fem.jsonl hp[0] − 0.48 (판), hg[0] − 0.48 (정답), mm.
 - r < 0.7 칸: sp/f*.npz rEnd (절 CM112 정의). 시작점 압축 되돌림 몫: 걸음 0 시작점 y0 에서 r < 0.98 인 칸 중 프레임 끝 r 이 y0 보다 0.005 넘게 큰 몫 (절 CM113 정의).
 - 걸음별 |g|/|g0|: sp.jsonl g2 (풀이기 잔차 r0 의 활성 노드 |r0|², 안 접은 것: 첫 평가 · 마지막 평가) -> sqrt(마지막/첫). fem.jsonl uf.g2 (풀이기 판정 값 ‖g‖²/F², 마지막 반복 시작, 접은 것) 도 함께."""
import json, os, sys
import numpy as np

O = sys.argv[1]
out = {}
for run in sys.argv[2:]:
    d = os.path.join(O, run)
    if not os.path.exists(os.path.join(d, "summary.json")):
        print(f"[{run}] summary 없음", flush=True); continue
    sm = json.load(open(os.path.join(d, "summary.json")))
    F = [json.loads(q) for q in open(os.path.join(d, "fem.jsonl")) if q.strip()]
    hp = {int(o["t"]) + 1: (1000 * (o["hp"][0] - 0.48), 1000 * (o["hg"][0] - 0.48)) for o in F if "hp" in o and "hg" in o}
    ufg = [x for o in F for x in ((o.get("uf") or {}).get("g2") or []) if x is not None]
    r07 = {}; rec = {}
    sp = os.path.join(d, "sp")
    for fn in sorted(os.listdir(sp)) if os.path.isdir(sp) else []:
        z = np.load(os.path.join(sp, fn)); f = int(fn[1:4])
        re_ = z["rEnd"].astype(np.float64); r07[f] = int((re_ < 0.7).sum())
        st = json.loads(str(z["steps"]))
        if st and "k0_r" in z.files:
            tags = [t[0] for t in st[0]["tags"]]
            if "y0_시작점" in tags:
                r0 = z["k0_r"][tags.index("y0_시작점")].astype(np.float64); m = r0 < 0.98
                if m.any():
                    rec[f] = [int(m.sum()), float((re_[m] > r0[m] + 0.005).mean())]
    ratio = []
    jl = os.path.join(d, "sp.jsonl")
    last = {}
    for q in open(jl) if os.path.exists(jl) else []:
        o = json.loads(q); last[o["fr"]] = o
    for f, o in sorted(last.items()):
        for a, b, c in (o.get("g2") or []):
            if a > 0 and b >= 0:
                ratio.append((f, float(np.sqrt(b / a)), int(c)))
    rr = np.array([q[1] for q in ratio]) if ratio else np.array([np.nan])
    recw = sum(v[0] * v[1] for v in rec.values()) / max(1, sum(v[0] for v in rec.values()))
    o = dict(err=sm.get("err_mean") if sm.get("err_mean") is not None else sm.get("err_mean_done"), diverged=sm.get("diverged"), frames_done=sm.get("frames_done"),
             ms_mean=sm.get("ms_mean"), it_nN=sm.get("it_nN"), it_nP=sm.get("it_nP"), h=hp, r07=r07, rec=rec, rec_w=recw,
             r07_f10_30=sum(v for f, v in r07.items() if 10 <= f <= 30), r07_f30=r07.get(30), r07_f50=r07.get(50),
             gratio=dict(n=len(ratio), med=float(np.nanmedian(rr)), p90=float(np.nanpercentile(rr, 90)), max=float(np.nanmax(rr)), frac_lt_1e2=float(np.mean(rr < 1e-2)), frac_lt_1e3=float(np.mean(rr < 1e-3)),
                         evals_med=float(np.median([q[2] for q in ratio])) if ratio else None),
             ufg2=dict(med=float(np.median(ufg)) if ufg else None, p90=float(np.percentile(ufg, 90)) if ufg else None), steps=ratio)
    out[run] = o
    hs = lambda f: (f"{hp[f][0]:.0f} (정답 {hp[f][1]:.0f})" if f in hp else "-")
    print(f"[{run}] 오차 {100 * (o['err'] or 0):.3f} % 발산 {o['diverged']} 끝 프레임 {o['frames_done']} 반복 {(o['it_nN'] or {}).get('mean')} PCG {(o['it_nP'] or {}).get('mean')} ms {o['ms_mean']}"
          f" | 높이 f30 {hs(30)} f50 {hs(50)} | r<0.7 f10~30 합 {o['r07_f10_30']} f30 {o['r07_f30']} f50 {o['r07_f50']} | 되돌림 몫 (가중) {recw:.2f}"
          f" | |g|/|g0| 중앙 {o['gratio']['med']:.2e} p90 {o['gratio']['p90']:.2e} <1e-2 {o['gratio']['frac_lt_1e2']:.2f} 평가 수 중앙 {o['gratio']['evals_med']} | uf.g2 중앙 {o['ufg2']['med']}", flush=True)
json.dump(out, open(os.path.join(O, "convabl_an.json"), "w"), ensure_ascii=False)
