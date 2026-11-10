"""절 CM98 묶음 condwhy (사용자 11-10 지시, 진단만 -- 결과 불변): H8 이 H5n 에 더한 세 항 (확장 T1E2 · P1X 강성 × κ (K) · S_구속·C) 중
무엇이 접촉 걸음 선형계의 조건수를 올리고 양의 정부호를 깨는지 가르기.

절 CM96 의 lib/anchorflow/fem_convwhy.py 를 그대로 씀 (저장 상태 다시 풀기·그래프 안 사본·랑초시). 바꾸는 것은 호스트 읽기 _read 하나:
- PCG 풀이마다 끊김 종류를 따로 셈 -- pAp ≤ 0 (헤시안 연산자 A 의 음의 곡률 방향: p 가 Ã 에서 0 이하) 과 rz ≤ 0 (전처리기 M 이 양정치 아님) 을 가름.
  첫 끊김 자리 (ci, rz, pAp, rz') 와 풀이 전체 (PCG 가 멈출 때까지) 의 pAp ≤ 0 · rz ≤ 0 · rz' ≤ 0 횟수.
- 랑초시 κ 는 절 CM96 과 같은 식 (첫 끊김 앞까지).
- pAp 상대 크기: 첫 끊김의 pAp / max_j pAp_j (반올림 크기 가늠; float32 곱이면 |pAp| 가 max 의 1e-7 근처면 반올림과 구별 안 됨).
"""
import math

from . import fem_convwhy as CW

_orig_read = CW._read


def _read(s, label, K):
    steps = _orig_read(s, label, K)
    nt = int(s.nt)
    T3 = s.cwT3.cpu().numpy()
    G, GT = s.cwG.cpu().numpy(), s.cwGT.cpu().numpy()
    for st in steps:
        k = st["k"]
        for o in st["its"]:
            if o.get("stop"):
                continue
            i = k * nt + o["it"]
            if T3[i, 0] < 0:
                continue
            npap = nrz = nrz1 = nrun = 0
            brk = None
            pmax = 0.0
            for ci in range(G.shape[1]):
                if GT[i, ci, 0] != 0 or GT[i, ci, 1] != 0:
                    break
                rz, pAp, rz1 = float(G[i, ci, 0]), float(G[i, ci, 2]), float(G[i, ci, 3])
                if rz == 0.0 and pAp == 0.0 and rz1 == 0.0:                               # 기록 없음 (상한 밖)
                    break
                nrun += 1
                pmax = max(pmax, abs(pAp))
                a, b, c = (not pAp > 0), (not rz > 0), (not rz1 > 0)
                npap += a; nrz += b; nrz1 += c
                if (a or b) and brk is None:
                    brk = dict(ci=ci, rz=rz, pAp=pAp, rz1=rz1, kind=("pAp" if a and not b else ("rz" if b and not a else "both")))
            if brk is not None:
                brk["pAp_rel"] = (brk["pAp"] / pmax) if pmax > 0 else None
            o.update(nrun=nrun, n_pAp=npap, n_rz=nrz, n_rz1=nrz1, brk=brk)
    return steps


def install(out, save="", load="", ks=1.0, nopl=0):
    CW._read = _read
    CW.install(out, save, load, ks, nopl)
    print("[절 CM98] condwhy 켬 -- PCG 끊김 종류 (pAp ≤ 0 · rz ≤ 0) 기록", flush=True)
