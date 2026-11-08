"""절 CM61 할 일 1: 레고 접촉 걸음의 몸 질량중심 법선 속도 -- 우리 판 (hc.jsonl 'mo', exe/hhtc2_wrap.py) 과 정답 MPM 궤적 (위치 차분) 대조.

사용: python exe/hhtc2_mom.py <정답 L000.pt> <판 디렉토리 ...>  -> markdown (stdout)
  정답: 입자 질량 같음 (MPM 채움) 으로 보고 질량중심 c_f = 평균 x_f; 걸음 평균 속도 v̄_f = (c_{f+1} − c_f)/h (위치 차분), 프레임 끝 속도는 v̄ 두 개의 평균 ((v̄_{f-1}+v̄_f)/2, 중심 차분).
  우리: 걸음 f (프레임 f -> f+1) 의 마지막 시도 줄 (K 걸음 중 마지막) -- V0 = P0/M (걸음 시작), VH = PH/M (HHT 끝), VB = PB/M (같은 y 후퇴 오일러 끝 = 걸음 평균).
  법선 = 바닥 법선 (레고 +z). H4 = 새 접촉 노드 법선 속도 합 뒤집힘 (절 CM58), H5 = H4 ∧ 몸 P0 < 0 < PH ∧ PB < 0, H5a = H4 ∧ 몸 P0 < 0 < PH (줄 'mo' 에서 다시 셈 -- 커밋 fc33f1c 의 셈 이름과 무관).
torch 가 없으면 zip 을 numpy 로 직접 읽음.
"""
import json
import os
import pickle
import sys
import zipfile

import numpy as np


def load_pt(path):
    try:
        import torch
        d = torch.load(path, map_location="cpu", weights_only=False)
        return {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in d.items()}
    except ImportError:
        pass
    DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32, "HalfStorage": np.float16,
          "ByteStorage": np.uint8, "BoolStorage": np.bool_}
    z = zipfile.ZipFile(path)
    pre = z.namelist()[0].split("/")[0]

    class S_:
        def __init__(s, n): s.n = n

    def rebuild(st, off, size, stride, *a):
        arr = st
        return np.lib.stride_tricks.as_strided(arr[off:], shape=tuple(size), strides=tuple(q * arr.itemsize for q in stride)).copy()

    class U(pickle.Unpickler):
        def find_class(s, mod, name):
            if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                return rebuild
            if name.endswith("Storage"):
                return S_(name)
            if mod.startswith("torch"):
                return lambda *a, **k: None
            return super().find_class(mod, name)

        def persistent_load(s, pid):
            _, cls, key, _, _ = pid
            return np.frombuffer(z.read(f"{pre}/data/{key}"), dtype=DT[cls.n])
    return U(z.open(f"{pre}/data.pkl")).load()


def gt_series(path, h=1.0 / 60.0, nrm=(0.0, 0.0, 1.0)):
    d = load_pt(path)
    x = np.asarray(d["x"], dtype=np.float64)
    c = x.mean(1) @ np.asarray(nrm)
    vb = np.diff(c) / h                                                                  # 걸음 f 평균 (f -> f+1)
    ve = np.full(len(c), np.nan)
    ve[1:-1] = 0.5 * (vb[:-1] + vb[1:])                                                  # 프레임 끝 (중심 차분)
    ve[0] = vb[0]
    return c, vb, ve


def ours(run):
    L = [json.loads(l) for l in open(os.path.join(run, "hc.jsonl")) if l.strip()]
    try:
        mode = json.load(open(os.path.join(run, "hcinfo.json")))["mode"]
    except Exception:
        mode = None                                                                       # 모드 0 의 napp 는 셈용 버림 표시라 갱신 바뀜이 아님
    out = {}
    for q in L:
        f = int(round(q["t"] / q["h"])) if q.get("h") else None
        mo = q.get("mo") or []
        if f is None or not mo:
            continue
        r = mo[-1]
        M = r[0]
        if not M > 0:
            continue
        out[f] = dict(V0=r[1] / M, VH=r[2] / M, VB=r[3] / M, K0=r[4], KH=r[5], KB=r[6], new=int(r[19]), S0=r[17], S1=r[18], Mn=r[9],
                      h4=int(q.get("nflip", 0) > 0), h5=int(q.get("nflip", 0) > 0 and r[1] < 0 < r[2] and r[3] < 0), h6=int(q.get("nflip", 0) > 0 and r[1] < 0 < r[2]), app=int(q.get("napp", 0) > 0 and mode in (1, 4, 5, 6)), nst=q["nstep"], K=q.get("K"),
                      M=M, rows=len(mo), anyc4=None)
    return out


def main():
    gt = sys.argv[1]
    runs = sys.argv[2:]
    c, vb, ve = gt_series(gt)
    R = {os.path.basename(r.rstrip("/")): ours(r) for r in runs}
    print("# 절 CM61 할 일 1 -- 레고 접촉 걸음 몸 질량중심 법선 속도 (m/s): 우리 판 대 정답 MPM (위치 차분)\n")
    print(__doc__.split("\n", 2)[2].split("torch")[0])
    # 정답 반전 걸음
    rev = [f for f in range(1, len(vb)) if ve[f] < 0 <= ve[f + 1]] if len(ve) > 2 else []
    print(f"정답 질량중심 법선 속도가 들어가는 쪽 -> 나오는 쪽으로 바뀌는 걸음 (프레임 끝 속도 기준): {['f%d' % f for f in rev]}\n")
    for nm, o in R.items():
        print(f"## {nm}\n")
        print("| 걸음 f | 정답 끝 v(f) | 정답 평균 v̄(f) | 정답 끝 v(f+1) | 우리 V0 | 우리 VB (평균) | 우리 VH (HHT 끝) | 받은 끝 | 새 접촉 노드 | H4 | H5 | H5a | 받은 갱신 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for f in sorted(o):
            q = o[f]
            if not (q["new"] > 0 or q["h4"] or q["h5"]):
                continue
            ve1 = ve[f + 1] if f + 1 < len(ve) else float("nan")
            vend = q["VB"] if q["app"] else q["VH"]
            print(f"| {f} | {ve[f]:+.3f} | {vb[f] if f < len(vb) else float('nan'):+.3f} | {ve1:+.3f} | {q['V0']:+.3f} | {q['VB']:+.3f} | {q['VH']:+.3f} | {vend:+.3f} | {q['new']} | "
                  f"{'예' if q['h4'] else ''} | {'예' if q['h5'] else ''} | {'예' if q['h6'] else ''} | {'후퇴 오일러' if q['app'] else 'HHT'} |")
        print()
    # 요약: H4 걸음 중 정답이 실제로 뒤집히는 걸음 (정답 v(f) < 0 ≤ v(f+1)) 몫, 우리 몸이 뒤집히는 몫
    print("## 요약 (판마다)\n")
    print("| 판 | H4 걸음 | 그중 정답 몸도 그 걸음에 뒤집힘 (v(f) < 0 ≤ v(f+1)) | 그중 우리 몸 HHT 뒤집힘 (V0 < 0 < VH) = H5a | 그중 걸음 평균 VB < 0 = H5 | H4 걸음 목록 | H5 걸음 목록 |")
    print("|---|---|---|---|---|---|---|")
    for nm, o in R.items():
        h4 = [f for f in sorted(o) if o[f]["h4"]]
        g = [f for f in h4 if f + 1 < len(ve) and ve[f] < 0 <= ve[f + 1]]
        b = [f for f in h4 if o[f]["V0"] < 0 < o[f]["VH"]]
        h5 = [f for f in h4 if o[f]["h5"]]
        print(f"| {nm} | {len(h4)} | {len(g)} {g} | {len(b)} {b} | {len(h5)} | {h4} | {h5} |")
    print()


if __name__ == "__main__":
    main()
