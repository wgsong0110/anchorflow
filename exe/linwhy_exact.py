"""절 CM127 linwhy 정확 해 대조 (인스턴스, scipy): 조립한 Ã (asm_*) 를 직접 풀어 x* -> 각 풀이 (M 원래 MG · I 전처리 없음 · De/Da Ã 블록 대각 · VE 층 0 Ã V 순환 · Mf 새 ω) 의
반복 k 해의 상대 에너지 노름 오차 ‖x−x*‖_Ã/‖x*‖_Ã, 질량 가중 변위 오차 ‖x−x*‖_m/‖x*‖_m, 상대 잔차. 사용: python exe/linwhy_exact.py <npz...>"""
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

for fn in sys.argv[1:]:
    z = np.load(fn)
    if "asm_r" not in z or "b_red" not in z:
        print(fn, "조립 없음"); continue
    n = z["asm_idx"].size
    A = sp.csr_matrix((z["asm_v"].astype(np.float64), (z["asm_r"], z["asm_c"])), shape=(n, n))
    A = ((A + A.T) * 0.5).tocsc()
    b = z["b_red"].astype(np.float64)
    xs = sla.spsolve(A, b)
    rr = np.linalg.norm(b - A @ xs) / np.linalg.norm(b)
    eAs = xs @ (A @ xs)
    nid = z["nid"]; idx = z["asm_idx"]; loc = np.searchsorted(nid, idx // 3)
    m = z["meff"][loc].astype(np.float64)
    nm2 = (m * xs ** 2).sum()
    print(f"{fn}: n {n} 직접 풀이 잔차 {rr:.1e}")
    keys = sorted({k.split('_')[1] for k in z.files if k.startswith('x_')})
    for nm in keys:
        ks = sorted(int(k.split('_')[2]) for k in z.files if k.startswith(f'x_{nm}_'))
        row = []
        for k in ks:
            x = z[f"x_{nm}_{k}"].astype(np.float64); d = x - xs
            row.append(f"{k}:{np.sqrt(max(d @ (A @ d), 0) / eAs):.2g}/{np.sqrt((m * d ** 2).sum() / nm2):.2g}/{np.linalg.norm(b - A @ x) / np.linalg.norm(b):.2g}")
        print(f"   {nm:4s} (에너지/질량/잔차) " + " ".join(row))
