"""manytraj: 다궤적 적합 입력 표준화의 스트리밍(Chan, float64) 합치기가 cat 후 mean·std(불편) 와 같은지 (AF_STATE_FIT_STATS_STREAM).
train_deform.py 의 식을 그대로 옮겨 상태마다 노드 수가 다른 무작위 특징(평균 큰 채널·상수 채널 포함)으로 대조한다."""
import torch

g = torch.Generator().manual_seed(0)
C = 33
chunks = []
for i in range(300):
    n = int(torch.randint(500, 3000, (1,), generator=g))
    f = torch.randn(n, C, generator=g) * torch.linspace(1e-3, 50, C) + torch.linspace(-100, 1e3, C)
    f[:, 5] = 2.5                                         # 상수 채널
    chunks.append(f.float())
f = torch.cat(chunks, 0)
sd = f.std(0); mu_ref = f.mean(0)
sd_ref = torch.where(sd > 1e-4 * sd.max().clamp(min=1e-12), sd, torch.ones_like(sd))
n_, mu_, m2_ = 0, None, None
for f_ in chunks:
    f_ = f_.double()
    nb = f_.shape[0]; mb = f_.mean(0); m2b = ((f_ - mb) ** 2).sum(0)
    if mu_ is None:
        n_, mu_, m2_ = nb, mb, m2b
    else:
        dl = mb - mu_; nt = n_ + nb
        mu_ = mu_ + dl * (nb / nt); m2_ = m2_ + m2b + dl ** 2 * (n_ * nb / nt); n_ = nt
s_ = (m2_ / max(n_ - 1, 1)).sqrt().float()
sd_st = torch.where(s_ > 1e-4 * s_.max().clamp(min=1e-12), s_, torch.ones_like(s_))
emu = float(((mu_.float() - mu_ref).abs() / mu_ref.abs().clamp(min=1e-6)).max())
esd = float(((sd_st - sd_ref).abs() / sd_ref).max())
# 정확한 값(float64 cat)과의 차도 같이: float32 cat 쪽 오차가 더 클 수 있다
f64 = f.double(); esd64 = float(((sd_st.double() - f64.std(0).where(sd > 1e-4 * sd.max(), torch.ones_like(f64.std(0)))).abs() / sd_ref.double()).max())
print(f"[스트리밍 표준화] 노드 표본 {n_} 채널 {C}: 평균 상대차 {emu:.2e}, 표준편차 상대차 (float32 cat 대비) {esd:.2e}, (float64 정확값 대비) {esd64:.2e}, "
      f"상수 채널 sd {float(sd_st[5])} (cat {float(sd_ref[5])})")
ok = emu < 1e-5 and esd < 1e-4 and float(sd_st[5]) == 1.0 and float(sd_ref[5]) == 1.0
print("STREAM_STATS_TEST_OK" if ok else "STREAM_STATS_TEST_FAIL")
