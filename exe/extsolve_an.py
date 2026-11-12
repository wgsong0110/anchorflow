"""절 CM117 집계 (CPU, 읽기만; exe/w5s0_an.py 복사, 출력 이름만): extsolve 50 프레임 판 -> <O>/extsolve_an.json.
사용: python exe/extsolve_an.py <O> <판> [<판> ...]
 - 오차·프레임 시간·뉴턴 반복: summary.json. 높이: fem.jsonl hp[0] (판 질량 가중 평균 높이, 절대) 와 hg[0] (정답 같은 프레임), 바닥 0.48 m 빼서 mm.
 - 칸 압축: sp/f*.npz rEnd (그 프레임 칸 두께 비, 절 CM112 정의) r < 0.7 칸 수 (프레임별, f10~30 합). 부분 구속 꼭짓점: k0_kbstat 최종."""
import json, os, sys
import numpy as np

O = sys.argv[1]
out = {}
for run in sys.argv[2:]:
    d = os.path.join(O, run)
    if not os.path.exists(os.path.join(d, "summary.json")):
        print(f"[{run}] summary.json 없음", flush=True); continue
    sm = json.load(open(os.path.join(d, "summary.json")))
    F = [json.loads(q) for q in open(os.path.join(d, "fem.jsonl")) if q.strip()]
    hp = {int(o["t"]): (1000 * (o["hp"][0] - 0.48), 1000 * (o["hg"][0] - 0.48)) for o in F if "hp" in o and "hg" in o}
    r07 = {}; part = {}; nc = {}
    for fn in sorted(os.listdir(os.path.join(d, "sp")) if os.path.isdir(os.path.join(d, "sp")) else []):
        z = np.load(os.path.join(d, "sp", fn)); f = int(fn[1:4])
        r = z["rEnd"].astype(np.float64)
        r07[f] = int((r < 0.7).sum()); nc[f] = int(len(r))
        if "k0_kbstat" in z.files:
            part[f] = json.loads(str(z["k0_kbstat"]))["KBS"].get("kb_최종", {}).get("corner_partial")
    o = dict(err=sm.get("err_mean"), diverged=sm.get("diverged"), frames_done=sm.get("frames_done"), ms_mean=sm.get("ms_mean"), ms_max=sm.get("ms_max"), ms_p90=sm.get("ms_p90"),
             it_nN=sm.get("it_nN", {}).get("mean"), h=hp, r07=r07, ncell=nc, partial=part,
             r07_f10_30=sum(v for f, v in r07.items() if 10 <= f <= 30), r07_max=max(r07.values()) if r07 else None)
    out[run] = o
    t = sorted(hp)
    print(f"[{run}] 오차 {100 * (o['err'] or 0):.3f} % 발산 {o['diverged']} 프레임 ms 평균 {o['ms_mean']:.2f} 최대 {o['ms_max']:.2f} 반복 {o['it_nN']}"
          + "".join(f" | f{q + 1} 높이 {hp[q][0]:.0f} mm (정답 {hp[q][1]:.0f})" for q in (29, 49) if q in hp)
          + f" | r<0.7 칸 f10~30 합 {o['r07_f10_30']} 최대 {o['r07_max']} | 부분 구속 꼭짓점 f12 {part.get(12)} f30 {part.get(30)} f50 {part.get(50)}", flush=True)
json.dump(out, open(os.path.join(O, "extsolve_an.json"), "w"), ensure_ascii=False)
