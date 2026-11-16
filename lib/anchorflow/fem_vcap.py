"""절 CM196 --be_sub_vcap c (실행기 exe/fem_an_rt_sim.py, --be_sub 위, 기본 0 = 이 모듈 안 부름): K > 1 걸음 끝 갱신 (rt_ga_end*) 바로 뒤에
노드 속도 크기를 에너지 상한 V_E 로 자름 (|v| > c V_E 인 노드만: v <- v c V_E/|v|, 걸음 변위를 맞춤 U += τ (v' − v) -- 후퇴 오일러 U1 = U0 + τ v1 와 같은 뜻, a = 0).
V_E = sqrt(max_p |v_p(0)|² + max |구속 속도|² + 2 |g| Δz) (Δz = 첫 프레임 입자의 중력 방향 높이 최대 − 바닥 높이; 바닥 없으면 V_E = max |v0| + |g| T).
근거 (절 CM193.5·CM194): N 64·128 실패는 질량이 평균의 1e-3 인 노드의 걸음 속도가 N 과 함께 커져 (N 32 10~23, N 64 17~37 m/s; wolf 정답 최대 4.47 m/s = V_E)
붙은 무응력 입자를 바닥 밖·아래로 끌고 나감. 외력이 중력뿐인 계에서 어느 물질 점도 V_E 보다 빠를 수 없으므로 (에너지 보존) 물리 노드에는 걸리지 않고, 풀이 오차로 생긴 속도만 자름.
입자는 쓰지 않음 (노드 속도·변위만)."""
import cupy as cp
import numpy as np
import torch

SRC = r"""
extern "C" {
__global__ void vc_cap(const int* IP, const unsigned char* fixd, float vmax, float dt, float* U, float* vk, float* ak, int* cnt) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9] || (fixd && fixd[n])) return;
    float v0 = vk[n*3], v1 = vk[n*3+1], v2 = vk[n*3+2];
    float s2 = v0*v0 + v1*v1 + v2*v2;
    if (!(s2 > vmax*vmax)) return;
    float r = vmax*rsqrtf(s2);
    for (int e=0;e<3;e++) { float v = vk[n*3+e]; float vc = v*r; U[n*3+e] += dt*(vc - v); vk[n*3+e] = vc; ak[n*3+e] = 0.f; }
    if (cnt) atomicAdd(cnt, 1);
}
}
"""
_M = {}


def _P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def vmax_energy(X0, V0, g, FL, T, extra_v=None):
    """에너지 상한 V_E (호스트에서 한 번, 첫 프레임 입자 상태로)."""
    g = np.asarray(g, dtype=np.float64); gn = float(np.linalg.norm(g))
    v2 = float((V0.double() ** 2).sum(1).max()) if V0 is not None and V0.numel() else 0.0
    if extra_v is not None:
        v2 = max(v2, float((extra_v.double() ** 2).sum(1).max()))
    if FL is None or gn == 0.0:
        return float(np.sqrt(v2) + gn * T), dict(kind="time", v0=float(np.sqrt(v2)), T=T)
    up = torch.tensor(-g / gn, device=X0.device, dtype=torch.float64)
    zt = float((X0.double() @ up).max())
    nr = np.asarray(FL["NR"], dtype=np.float64); pt = np.asarray(FL["PT"], dtype=np.float64)
    zf = float(pt @ (-g / gn))                                                         # 바닥 점의 높이 (중력 반대 방향)
    dz = max(zt - zf, 0.0)
    return float(np.sqrt(v2 + 2.0 * gn * dz)), dict(kind="energy", v0=float(np.sqrt(v2)), dz=dz, g=gn, nr=nr.tolist())


def apply(s, vmax, dt):
    """그래프 캡처 중에도 부를 수 있음 (커널 하나, 동기화 없음)."""
    from . import fem_rt_cuda as RTC
    if "m" not in _M:
        _M["m"] = cp.RawModule(code=SRC); _M["f"] = _M["m"].get_function("vc_cap")
    if getattr(s, "_vc_cnt", None) is None:
        s._vc_cnt = torch.zeros(1, device=s.x.device, dtype=torch.int32)
    g = ((int(s.capN) + 255) // 256,)
    with RTC._Stream():
        _M["f"](g, (256,), (_P(s.IP), _P(s.fixd), np.float32(vmax), np.float32(dt), _P(s.U), _P(s.vk), _P(s.ak), _P(s._vc_cnt)))
