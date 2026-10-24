"""--apic 실행 점검 (사용자 10-24): 프레임 파일에서 (b) C 갱신이 F 갱신 야코비안과 일관, (d) 덤프·이어 돌리기에 C.

  python exe/apic_check.py --frames <실행>/frames [--ref <끊지 않은 실행>/frames] [--dump <실행>/roll_net.pt]
  (b) 탄성(jelly)이면 F_out = J F_in, C_out = (J - I)/h  =>  F_out = (I + h C_out) F_in  (h = frame_dt)
  (d) 프레임 t+1 의 state_in.C == 프레임 t 의 state_out.C (이어 돌기에서 C 를 읽어 온 것 포함), roll_net.pt 의 C == 프레임 파일 C,
      --ref: 끊었다 이어 돈 실행과 끊지 않은 실행의 같은 프레임 상태 차
"""
import argparse
import glob
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--frames", required=True)
ap.add_argument("--ref", default=None)
ap.add_argument("--dump", default=None)
a = ap.parse_args()
OK = [True]


def chk(name, ok, msg=""):
    print(f"[{'통과' if ok else '실패'}] {name} {msg}", flush=True)
    OK[0] &= bool(ok)


def load(fd):
    fs = sorted(glob.glob(os.path.join(fd, "frame_*.pt")))
    return [torch.load(f, map_location="cpu", weights_only=False) for f in fs]


fr = load(a.frames)
print(f"프레임 파일 {len(fr)} 개 ({a.frames})")
I3 = torch.eye(3, dtype=torch.float64)
for c in fr:
    t, si, so = c["t"], c["state_in"], c["state_out"]
    chk(f"프레임 {t}: state_in·state_out 에 C", ("C" in si) and ("C" in so))
    h = float(c["frame_dt"])
    Fi, Fo, Co = si["F"].double(), so["F"].double(), so["C"].double()
    pred = (I3 + h * Co) @ Fi
    e = float((pred - Fo).abs().max())
    chk(f"프레임 {t}: (b) F_out = (I + h C_out) F_in", e < 1e-4, f"최대 차 {e:.2e} (|F-I| 최대 {float((Fo - I3).abs().max()):.3e}, |C| 최대 {float(Co.abs().max()):.3e}, "
        f"|C 반대칭| 평균 {float((0.5 * (Co - Co.transpose(-1, -2))).reshape(-1, 9).norm(dim=-1).mean()):.3e})")
    if t == 0:
        chk("프레임 0: 시작 C = 0", float(si["C"].abs().max()) == 0.0)
for c0, c1 in zip(fr[:-1], fr[1:]):
    e = float((c1["state_in"]["C"] - c0["state_out"]["C"]).abs().max())
    chk(f"(d) 프레임 {c1['t']} 입력 C = 프레임 {c0['t']} 출력 C", e == 0.0, f"{e:.2e}")
if a.dump:
    d = torch.load(a.dump, map_location="cpu", weights_only=False)
    chk("(d) roll_net.pt 에 C", "C" in d and d["C"].shape[0] == len(fr), str(tuple(d["C"].shape)) if "C" in d else "없음")
    if "C" in d:
        e = max(float((d["C"][c["t"]] - c["state_out"]["C"]).abs().max()) for c in fr)
        chk("(d) roll_net.pt C == 프레임 파일 C", e == 0.0, f"{e:.2e}")
if a.ref:
    rf = {c["t"]: c for c in load(a.ref)}
    for c in fr:
        r = rf.get(c["t"])
        if r is None:
            continue
        dx = float((c["state_out"]["x"] - r["state_out"]["x"]).norm(dim=-1).max())
        dC = float((c["state_out"]["C"] - r["state_out"]["C"]).abs().max())
        ext = float(c.get("EXT", 1.0))
        print(f"    [끊고 이어 돈 실행 vs 끊지 않은 실행] 프레임 {c['t']}: |Δx| 최대 {dx:.3e} ({100 * dx / ext:.4f}% EXT), |ΔC| 최대 {dC:.3e} "
              f"(|C| 최대 {float(r['state_out']['C'].abs().max()):.3e}), E {c['rec'].get('E_ip'):.6e} vs {r['rec'].get('E_ip'):.6e}", flush=True)
print("APIC_CHECK OK" if OK[0] else "APIC_CHECK FAIL", flush=True)
