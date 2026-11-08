"""절 CM43 묶음 escape2 고침 후보 (수박 옛 실행기 lib/anchorflow/fem_ao_rt.py 만; 인자 없으면 이 모듈을 안 부름 = 소스·그래프 그대로).

원인 (절 CM43.1, 진단 D2): 이탈은 '고정 (m_n < mthr) 이 아닌 가벼운 노드' (질량 비 1e-3 ~ 2e-2) 하나가 한 걸음에 12 ~ 128 m/s 를 받는 데서 시작
-> 그 노드를 27 노드에 둔 입자 (가중 0.01 ~ 0.07) 가 몇 m/s 를 받고 logJp 가 한 걸음에 3 ~ 8 로 뛰어 (응력 0 의 파편) 그대로 날아감.

M1o (--e2mf κ): 뉴턴 풀이가 쓰는 노드 관성 질량에 노드별 하한 m'_n = max(m_n, κ M / (8 n_c)) (m_n > 0 노드만; M = 입자 질량 합, n_c = 이 프레임 찬 칸 수 IP[11])
  = 절 CM42 후보 M1 (새 실행기 rt_meff) 과 같은 정의를 옛 실행기에. 바꾸는 곳 = 관성 몫을 쓰는 풀이 커널의 질량 인자만:
    rt_grad(_c) (기울기 Mi η·블록 야코비 대각), rt_ap(_c) (헤시안 곱), ao_warm (첫 추정), en_phi_n (절 CM33 반복 되돌림 에너지).
  그대로: 노드 질량·운동량 (mn·mv -> 속도 vk = mv/mn, 관성 예측 ŷ), 고정 판정 (ao_floor_n 의 m_n < mthr), 걸음 끝 갱신, G2P, 진단.
  이웃 노드를 쓰지 않음. κ 는 다른 네 씬 공통 설정의 --gq_oct 0.1 과 같은 값 (점유 가중 판 가우스 점 질량 하한 m_q ≥ κ m_c / 8 을 평균 칸 질량으로 쓴 값).
"""
import numpy as np
import torch

SOLVE = ("rt_grad", "rt_grad_c", "rt_ap", "rt_ap_c", "ao_warm", "en_phi_n")


def install(cand, out):
    from . import fem_ao_rt as AORT
    from . import fem_an_rt as ART
    kap = float(cand.get("e2mf", 0.0))
    if kap <= 0:
        return
    P = ART.P
    _ok = AORT.AORTSolver.k

    def k(s, name, n, args, b=128):
        if name == "ao_act" and getattr(s, "_e2km", None) is not None:              # P2G·노드 속도 뒤, 걸음 앞 (노드 질량이 이 프레임 값으로 다 찬 뒤) 한 번
            _ok(s, name, n, args, b)
            if getattr(s, "e2meff", None) is None or s.e2meff.shape != s.mn.shape:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("--e2mf: meff 버퍼는 데움 record 에서 잡아야 함")
                s.e2meff = torch.zeros_like(s.mn)
            fl = s._e2km / (8.0 * s.IP[11:12].float().clamp_min(1.0))               # 장치 위 (호스트 읽기 없음 -- 그래프 안)
            torch.where(s.mn > 0, torch.maximum(s.mn, fl), s.mn, out=s.e2meff)
            return
        if name in SOLVE and getattr(s, "_e2km", None) is not None:
            pm, pe = P(s.mn), P(s.e2meff)
            args = tuple(pe if (isinstance(a, np.uint64) and a == pm) else a for a in args)
        return _ok(s, name, n, args, b)
    AORT.AORTSolver.k = k
    _oi = AORT.AORTSolver.alloc

    def alloc(s, need):
        r = _oi(s, need)
        if getattr(s, "_e2km", None) is None:
            s._e2km = float(kap * float(s.m.double().sum()))
            print(f"[절 CM43 후보 M1o] 풀이 관성 질량 하한 κ {kap} x 질량 합 {s._e2km / kap:.4g} kg / (8 x 찬 칸 수) (옛 실행기)", flush=True)
        s.e2meff = torch.zeros_like(s.mn)
        return r
    AORT.AORTSolver.alloc = alloc
