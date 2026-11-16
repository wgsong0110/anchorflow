"""절 CM193 --be_sub_lv c (실행기 exe/fem_an_rt_sim.py, --be_sub 위, 기본 0 = 이 모듈 안 부름): K > 1 걸음 끝 갱신 (rt_ga_end*) 바로 뒤에
'가장자리 노드' (질량 0 < m_n < c x 27 이웃 최대 질량) 의 걸음 속도를 27 이웃 질량 가중 평균 속도로 바꾸고 그 걸음 변위를 맞춤
(U_n += τ (v̄ − v_n), v_n = v̄, a_n = 0) -- MPM 격자에서 노드 속도 = 둘레 입자 운동량/질량 인 것과 같은 뜻 (거의 빈 노드는 자기 운동이 아니라 이웃 물질의 속도).
근거 (절 CM193.5): N 64·128 실패는 질량이 평균의 1e-3 인 노드의 걸음 속도가 N 과 함께 커져 (N 32 10~23, N 64 17~37 m/s) 붙은 입자를 바닥 밖·아래로 끌고 나감."""
import cupy as cp
import numpy as np
import torch

SRC = r"""
extern "C" {
__global__ void lv_avg(const int* IP, const float* mn, const float* vk, float c, float* vb, int* fl) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    fl[n] = 0;
    float m = mn[n];
    if (!(m > 0.f)) return;
    int nb0 = IP[6], nb1 = IP[7], nb2 = IP[8];
    int i = n / (nb1*nb2), j = (n / nb2) % nb1, k = n % nb2;
    float mm = 0.f, sm = 0.f, s[3] = {0.f, 0.f, 0.f};
    for (int a=-1;a<=1;a++) for (int b=-1;b<=1;b++) for (int d=-1;d<=1;d++) {
        int ii = i+a, jj = j+b, kk = k+d;
        if (ii < 0 || jj < 0 || kk < 0 || ii >= nb0 || jj >= nb1 || kk >= nb2) continue;
        int q = (ii*nb1 + jj)*nb2 + kk; float w = mn[q];
        if (!(w > 0.f)) continue;
        mm = fmaxf(mm, w); sm += w; for (int e=0;e<3;e++) s[e] += w*vk[q*3+e];
    }
    if (m < c*mm && sm > 0.f) { fl[n] = 1; for (int e=0;e<3;e++) vb[n*3+e] = s[e]/sm; }
}
__global__ void lv_set(const int* IP, const int* fl, const float* vb, const unsigned char* fixd, float dt, float* U, float* vk, float* ak) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || !fl[n] || (fixd && fixd[n])) return;
    for (int e=0;e<3;e++) { float v = vb[n*3+e]; U[n*3+e] += dt*(v - vk[n*3+e]); vk[n*3+e] = v; ak[n*3+e] = 0.f; }
}
}
"""
_M = {}


def _P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def apply(s, c, dt):
    """그래프 캡처 중에도 부를 수 있음 (커널 두 개, 동기화 없음). 버퍼는 용량 바뀌면 다시."""
    from . import fem_rt_cuda as RTC
    if "m" not in _M:
        _M["m"] = cp.RawModule(code=SRC); _M["a"] = _M["m"].get_function("lv_avg"); _M["s"] = _M["m"].get_function("lv_set")
    if getattr(s, "_lv_cap", None) != s.capN:
        s._lv_vb = torch.zeros(s.capN, 3, device=s.x.device, dtype=torch.float32); s._lv_fl = torch.zeros(s.capN, device=s.x.device, dtype=torch.int32); s._lv_cap = s.capN
    g = ((int(s.capN) + 255) // 256,)
    with RTC._Stream():
        _M["a"](g, (256,), (_P(s.IP), _P(s.mn), _P(s.vk), np.float32(c), _P(s._lv_vb), _P(s._lv_fl)))
        _M["s"](g, (256,), (_P(s.IP), _P(s._lv_fl), _P(s._lv_vb), _P(s.fixd), np.float32(dt), _P(s.U), _P(s.vk), _P(s.ak)))
