"""절 CM35 묶음 common2 (조정자 위임 11-08): 공통 후보 B -- 수박 (c)+(d) (절 CM33) 와 레고를 지키는 --ue_loc (절 CM31) 를 한 세팅으로.

정의 (끄면 생성 CUDA 소스 글자 그대로 -- 훅은 플래그 켤 때만 소스·커널 열을 바꿈):
- 새 실행기 (fem_an_rt; 레고·wolf·ficus·bread) 에 절 CM33 (c) `--xit_from j` (`--newton N` 위): 반복 j 시작 잔차를 잰 (cj_g) 뒤 그때까지 덜 풀림 판정
  (걸음 보폭 최소 < 1 또는 이 반복 시작 ‖g‖ > F) 이 아니면 그 걸음 나머지 반복을 건너뜀 (허용치 멈춤과 같은 CT[0] = 1·AF_SKIP). 옛 실행기 en_gate 와 같은 식.
  뉴턴 상한 = 덜 풀린 걸음 N, 아닌 걸음 j (허용치로 먼저 멈추면 그 전).
- 새 실행기 `--en_mo` (절 CM33 (d)): 새 실행기 선탐색 (--ls_n, 아르미호 E(c) ≤ E0 + 1e-4 c g·δ, 후보 2^0..2^-(ls_n−1), 없으면 보폭 0) 이 이미
  '반복 갱신이 증분 포텐셜 Φ = 관성 + 탄성 + 장벽 을 올리면 반씩 줄이고 그래도 오르면 버림' 을 (더 엄격하게) 함 -> 소스·커널 열 그대로, 선탐색 판인지만 확인.
- `--ue_dil R` (--ue_loc 위, 두 실행기): 덜 풀린 걸음 후퇴 오일러 노드 집합 S 를 27 이웃으로 R 번 넓힘 (노드 상자 격자, 걸음마다).
- `--ue_loc` 비트 16 (두 실행기, 절 CM35 진단이 가리킨 것): 덜 풀린 걸음 끝 HHT 갱신 속도 v1 = al (y − z0) 가 걸음 시작 속도와 반대 (v1·v0 < 0) 인 노드 --
  수박 첫 접촉 (f4) 에서 S 밖 노드는 자른·안 자른 모두 ρ = −0.7~−0.83 으로 뒤집힘 (절 CM23.4 기제) -> 뒤집힐 노드만 후퇴 오일러로.
- `--ue_loc` 비트 8 (옛 실행기만, --ws_bar 판 + 절 CM33 en 버퍼): 이 걸음 어느 뉴턴 반복이든 노드별 국소 보폭으로 자른 (k > 0) 노드.
  새 실행기는 노드별 보폭이 없어 (보폭은 몸 전체 스칼라) 비트 8 은 빈 집합 -- 받기만 함.
진단 (결과 불변, 옛 실행기 --enrec 위): 노드 집합 S / S 밖 · 자른 노드 나눔 몫과 HHT 속도 뒤집기 척도 (split_s).
"""
import numpy as np
import torch

C2_SRC = r"""
// ---- 절 CM35 (lib/anchorflow/fem_common2.py) ----
__global__ void ul_dil(const int* IP, const int* src, int* dst) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    if (src[n]) { dst[n] = 1; return; }
    int n0 = IP[6], n1 = IP[7], n2 = IP[8];
    int i = n/(n1*n2), j = (n/n2) % n1, k = n % n2;
    int r = 0;
    for (int a=-1;a<=1 && !r;a++) for (int b=-1;b<=1 && !r;b++) for (int c=-1;c<=1 && !r;c++) {
        int ii = i+a, jj = j+b, kk = k+c;
        if (ii < 0 || jj < 0 || kk < 0 || ii >= n0 || jj >= n1 || kk >= n2) continue;
        if (src[(ii*n1 + jj)*n2 + kk]) r = 1;
    }
    dst[n] = r;
}
__global__ void ul_cpy(const int* IP, const int* src, int* dst) { int n = blockIdx.x*blockDim.x + threadIdx.x; if (n < IP[9]) dst[n] = src[n]; }
// 비트 16: 덜 풀린 걸음 끝 HHT 갱신이 노드 속도를 뒤집는 노드 (v1_HHT = al (y − z0), v1_HHT·v0 < 0) -- 두 실행기 걸음 끝 커널과 같은 v1 식, 갱신 전 vk = v0
__global__ void ul_flip(const int* IP, float al, const float* y, const float* z0, const float* vk, int* m) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= IP[9]) return;
    float d = 0.f, s = 0.f;
    for (int a=0;a<3;a++) { float v0 = vk[n*3+a]; d += al*(y[n*3+a] - z0[n*3+a])*v0; s += v0*v0; }
    if (s > 0.f && d < 0.f) m[n] = 1;
}
__global__ void ul_cut(const int* IP, const int* cut, int* m) { int n = blockIdx.x*blockDim.x + threadIdx.x; if (n < IP[9] && cut[n] > 0) m[n] = 1; }
"""

C2_GATE_SRC = r"""
__global__ void c2_gate(int* CT, const float* cjs, const float* cjg, int k, int res) {
    if (CT[0]) return;
    bool cut = cjs[k] < 1.f, big = res && cjg[k] > 1.f;
    if (!(cut || big)) { CT[0] = 1; AF_SKIP = 1; }
}
"""


def c2ify(src, dil, gate):
    """모듈 끝에 커널 (끄면 소스 그대로). dil = --ue_dil·비트 8 커널, gate = 새 실행기 --xit_from."""
    if not (dil or gate):
        return src
    i = src.rstrip().rfind("}")
    return src[:i] + (C2_SRC if dil else "") + (C2_GATE_SRC if gate else "") + "\n}\n"


def dilate(s, capN, IP, R):
    """S (s.ulm) 를 27 이웃으로 R 번 넓힘 (s.ulm2 사본)."""
    for _ in range(int(R)):
        s.k("ul_dil", capN, (_P(IP), _P(s.ulm), _P(s.ulm2)), b=256)
        s.k("ul_cpy", capN, (_P(IP), _P(s.ulm2), _P(s.ulm)), b=256)


def _P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


# ======================= 진단 (옛 실행기 --enrec 위, 결과 불변) =======================
def split_s(S, m, has, v0, v1, y, U1, dt, free, cut, pr, pu, prr, gph):
    """S = 걸음 끝 후퇴 오일러 노드 집합 (ulm), 나눔: S · S 밖 · S 밖 ∩ 자른 노드 · S 밖 ∩ 안 자른 노드.
    척도: 질량, 운동 에너지 앞·뒤, 속도 비 ρ = Σ m v1·v0 / Σ m |v0|² (HHT 로 v1 이 v0 를 뒤집으면 < 0), 뒤집힌 질량 몫 (v1·v0 < 0, |v0| > 1e-4),
    후퇴 오일러였다면의 운동 에너지 ½ Σ m |y/dt|² (K 1 이라 걸음 시작 U = 0), 잔차 일·갱신식 몫·불일치·수치 감쇠."""
    ulm = getattr(S, "ulm", None)
    nn = m.shape[0]
    Sm = (ulm[:nn] > 0) & has if ulm is not None else torch.zeros_like(has)
    vbe = y / dt
    d = (0.5 * m)[:, None]
    out = {}
    for nm_, R in (("S", Sm), ("nS", has & ~Sm), ("nS_cut", has & ~Sm & cut), ("nS_unc", has & ~Sm & ~cut)):
        mm = m[R]
        dv0 = (v0[R] ** 2).sum(1)
        dot = (v1[R] * v0[R]).sum(1)
        flip = (dot < 0) & (dv0.sqrt() > 1e-4)
        out[nm_] = dict(n=int(R.sum()), m=float(mm.sum()), KE0=float((0.5 * mm * dv0).sum()), KE1=float((0.5 * mm * (v1[R] ** 2).sum(1)).sum()),
                        KEbe=float((d[R][:, 0] * (vbe[R] ** 2).sum(1)).sum()),
                        rho=float((mm * dot).sum() / max(float((mm * dv0).sum()), 1e-30)), mflip=float(mm[flip].sum()) / max(float(mm.sum()), 1e-30),
                        vz0=float((mm * v0[R][:, 2]).sum()) / max(float(mm.sum()), 1e-30), vz1=float((mm * v1[R][:, 2]).sum()) / max(float(mm.sum()), 1e-30),
                        Wres=float(pr.sum(1)[R & free].sum()), Rupd=float(prr.sum(1)[R].sum()), Wupd=float(pu.sum(1)[R].sum()),
                        Dnum=float((0.5 * mm * ((v1[R] - v0[R]) ** 2).sum(1)).sum()))
    return out
