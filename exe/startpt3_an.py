"""절 CM115 집계 (CPU, 읽기만): <판>/sp/f*.npz 의 k0_kbstat -> <O>/startpt3_<판>.json (+ 줄 출력).
구속 몫 정의별 (kb_기본 = ao_pbc_k e^-x, kb_S = S_구속 덮어쓰기, kb_최종 = 걸음에 쓰인 값) 부분 구속 (1e-3 < k ≤ 0.999) 노드·칸 꼭짓점 수,
그중 확장 외부 노드·방출 앞면 노드 (받침 입자 중 일부만 미방출) 수, 노드 받침 입자 수 중앙."""
import json, os, sys
import numpy as np

O = sys.argv[1]
for run in sys.argv[2:]:
    d = os.path.join(O, run, "sp")
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".npz"):
            continue
        z = np.load(os.path.join(d, fn))
        if "k0_kbstat" not in z.files:
            continue
        f = int(fn[1:4]); st = json.loads(str(z["k0_kbstat"]))
        if "k0_kb_기본" in z.files and "k0_kb_최종" in z.files:
            a, b = z["k0_kb_기본"].astype(np.float64), z["k0_kb_최종"].astype(np.float64)
            st["corner_basic_vs_final_maxdiff"] = float(np.abs(a - b).max())
        if "k0_kb_S" in z.files and "k0_kb_최종" in z.files:
            st["corner_S_vs_final_maxdiff"] = float(np.abs(z["k0_kb_S"].astype(np.float64) - z["k0_kb_최종"].astype(np.float64)).max())
        out[f] = st
        print(f"[{run}] f{f}: 활성 {st['nact']} 앞면 {st['front']} (꼭짓점 {st['front_corner']}) 확장 {st['ext']} 받침 입자 중앙 {st['cnt_med']} 앞면 {st['cnt_front_med']} 미방출 {st['held_part']:.3f} | "
              + " | ".join(f"{k}: 부분 {v['partial']} (꼭짓점 {v['corner_partial']}, 확장 {v['partial_ext']}, 앞면 {v['partial_front']}) 묶임 {v['held']} 풀림 {v['free']}" for k, v in st["KBS"].items())
              + f" | 차 기본-최종 {st.get('corner_basic_vs_final_maxdiff')} S-최종 {st.get('corner_S_vs_final_maxdiff')}", flush=True)
    json.dump(out, open(os.path.join(O, f"startpt3_{run}.json"), "w"), ensure_ascii=False)
