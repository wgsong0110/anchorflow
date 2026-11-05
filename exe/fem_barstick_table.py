"""절 CM12 표 (barstick): python exe/fem_barstick_table.py <묶음 디렉토리 barstick> <barx 디렉토리 (x_wolf·o_wolf·o_L000·o_drop1)> <정답 wolf.pt>
판: s_<씬> = (B) + --fl_stk (장벽 + (A) 가우스 점 고착 함께), o_<씬> = (B) 그대로 (이 인스턴스, 새 코드 플래그 없음 -- 끄면 같은지),
a_wolf = (A) 가우스 점 고착 (절 CM8 a3 명령, 상한 PCG 3) 다시 실행 (높이 곡선·영상용), (B) 높이 곡선·덤프 = barx x_wolf (= barx o_wolf 비트 같음).
높이 = 바닥 법선 (z), 질량 중심 z 는 질량 가중 (실행기 --hgt), 퍼짐 폭 = 바닥 평면 투영 x·y 각각 1~99 백분위 범위 중 큰 값 (exe/fem_lift_table.py 와 같은 정의)."""
import json
import os
import re
import sys

import numpy as np

B, BX, GTP = sys.argv[1:4]


def sm(d):
    p = os.path.join(d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def jl(d):
    p = os.path.join(d, "fem.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def state(d):
    s = sm(d)
    if s is None:
        ri = os.path.join(d, "run_info.txt")
        t = open(ri).read() if os.path.exists(ri) else ""
        return ("실패 (" + t.split("rc=")[-1].split()[0] + ")") if "rc=" in t else ("없음" if not t else "도는 중")
    dv = s.get("diverged")
    e = 100 * (s["err_mean_done"] if dv else s["err_mean"])
    return ("**x%d** (발산 전 %.2f)" % (dv, e)) if dv else "%.2f" % e


def drop(d):
    q = os.path.join(d, "drop_ana.txt")
    if not os.path.exists(q):
        return ""
    m = re.search(r"\n[^\n]*?:[^\n]*반발 계수 ([0-9.\-]+), 튀어 오름 ([0-9.\-]+) mm", "\n" + "\n".join(l for l in open(q).read().splitlines() if not l.startswith("mpm:")))
    return (" / 반발 %s · 튀어 오름 %s mm" % m.groups()) if m else ""


H = lambda d: (sm(d) or {}).get("x_sha1", "")
print("# 절 CM12 barstick -- 장벽 바닥 (B) (장벽 + --y0eb, 절 CM8 b 판 명령) 에 (A) 가우스 점 고착 (절 CM8 a 판 정의) 을 함께 켠 판 (--fl_stk), 결정적")
print()
print("## 1. 판정 (칸 = 평균 오차 % 또는 x발산 프레임; 낙하 정답 반발 0.750 · 튀어 오름 83.4 mm)")
print()
print("| 씬 | (B) o (이 인스턴스) | (B) barx o (절 CM10) | (B)+고착 s | 해시 o / barx o / s |")
print("|---|---|---|---|---|")
for t in ("wolf", "L000", "0016", "drop1"):
    o, bo, s = os.path.join(B, "o_" + t), os.path.join(BX, "o_" + t), os.path.join(B, "s_" + t)
    c = [state(o) + (drop(o) if t == "drop1" else ""), (state(bo) + (drop(os.path.join(BX, "o_drop1")) if t == "drop1" else "")) if sm(bo) else "-",
         state(s) + (drop(s) if t == "drop1" else "")]
    print(f"| {t} | " + " | ".join(c) + f" | {H(o)} / {H(bo) or '-'} / {H(s)} |")
a = os.path.join(B, "a_wolf")
print(f"| wolf (A) a (상한 PCG 3, 다시 실행) | {state(a)} | 절 CM8 a3 3.79 (클러스터) | | {H(a)} |")
print()
print("- 끄면 같은지: o (새 코드, --fl_stk 없음) 해시 = barx o (절 CM10 인스턴스, 그 때 코드) 이면 같은 기계어·같은 결과 (다른 vast 기계면 NVRTC 환경 차로 다를 수 있음 -- 절 CM10.2).")
print()
JA, JB, JS = jl(a), jl(os.path.join(BX, "x_wolf")), jl(os.path.join(B, "s_wolf"))
if not (JS and JB):
    sys.exit(0)
def tload(path):
    """torch 없는 곳에서도 .pt (zip) 의 텐서를 numpy 로 (float32·float64·int 저장소만)."""
    try:
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    except ImportError:
        pass
    import pickle
    import zipfile
    z = zipfile.ZipFile(path)
    pre = z.namelist()[0].split("/")[0]
    DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32, "HalfStorage": np.float16, "BoolStorage": np.bool_, "ByteStorage": np.uint8}

    class St:
        def __init__(s, key, dt):
            s.key, s.dt = key, dt

    def rebuild(st, off, size, stride, *a):
        buf = np.frombuffer(z.read(f"{pre}/data/{st.key}"), dtype=st.dt)
        it = buf.itemsize
        return np.lib.stride_tricks.as_strided(buf[off:], shape=tuple(size), strides=tuple(q * it for q in stride)).copy()

    class Up(pickle.Unpickler):
        def find_class(s, mod, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if mod == "torch" and name.endswith("Storage"):
                return DT[name]
            if mod == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            return lambda *a, **k: None

        def persistent_load(s, pid):
            return St(pid[2], pid[1] if not isinstance(pid[1], type(None)) else np.float32)

    return Up(z.open(f"{pre}/data.pkl")).load()


class _N:
    def __init__(s, a):
        s.a = a
        s.shape = a.shape

    def __getitem__(s, k):
        return _N(s.a[k])

    def numpy(s):
        return np.asarray(s.a)


def _np(x):
    return x if hasattr(x, "numpy") else _N(np.asarray(x))


try:
    G = _np(tload(GTP)["x"])
    pr = lambda d: _np(tload(os.path.join(d, "pred.pt"))["pred"]) if os.path.exists(os.path.join(d, "pred.pt")) else None
    SD = os.path.join(B, "s_wolf_dp") if os.path.exists(os.path.join(B, "s_wolf_dp", "pred.pt")) else os.path.join(B, "s_wolf")
    PA, PB, PS = pr(a), pr(os.path.join(BX, "x_wolf")), pr(SD)
except Exception as ex:
    print(f"(퍼짐 폭 생략: {ex})")
    G = PA = PB = PS = None


def spread(x):
    q = np.percentile(x[:, :2], [1, 99], axis=0)
    return float((q[1] - q[0]).max())


f = lambda v: ("%.1f" % (1000 * v)) if v is not None and np.isfinite(v) else "-"
pc = lambda e: "%.2f" % (100 * e) if e is not None else "-"
n = max(len(JS), len(JA), len(JB))
rows = []
for t in range(n):
    g = lambda J, k: (J[t]["hp"][k] if t < len(J) and "hp" in J[t] else None)
    sp = lambda P: spread(P[t].numpy()) if P is not None and t < P.shape[0] else float("nan")
    hg = (JS[t] if t < len(JS) else (JB[t] if t < len(JB) else JA[t]))["hg"]
    rows.append(dict(fr=t + 1, hg=hg, c=[g(JA, 0), g(JB, 0), g(JS, 0)], m=[g(JA, 1), g(JB, 1), g(JS, 1)],
                     w=[sp(G[1:]) if G is not None else float("nan"), sp(PA), sp(PB), sp(PS)],
                     e=[JA[t].get("err") if t < len(JA) else None, JB[t].get("err") if t < len(JB) else None, JS[t].get("err") if t < len(JS) else None],
                     bc=JS[t].get("bc", {}).get("neng") if t < len(JS) else None))
for title, sel in (("2. wolf 곡선 요약 (프레임 10·20·30·40·50)", lambda R: R["fr"] % 10 == 0), ("3. wolf 곡선 프레임별", lambda R: True)):
    print(f"## {title}")
    print()
    print("| 프레임 | 질량중심 z 정답 | (A) | (B) | (B)+고착 | 최대 z 정답 | (A) | (B) | (B)+고착 | 퍼짐 폭 정답 | (A) | (B) | (B)+고착 | 오차 % (A) | (B) | (B)+고착 | 장벽 걸린 점 (B)+고착 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for R in filter(sel, rows):
        print(f"| {R['fr']} | {f(R['hg'][0])} | " + " | ".join(f(v) for v in R["c"]) + f" | {f(R['hg'][1])} | " + " | ".join(f(v) for v in R["m"]) + " | "
              + " | ".join(f(v) for v in R["w"]) + " | " + " | ".join(pc(e) for e in R["e"]) + f" | {R['bc'] if R['bc'] is not None else '-'} |")
    print()
for nm, d in (("(A) a_wolf", a), ("(B) barx x_wolf", os.path.join(BX, "x_wolf")), ("(B)+고착 s_wolf", os.path.join(B, "s_wolf"))):
    J = jl(d)
    if J:
        nN = np.mean([q["it"]["nN"] for q in J]); nP = np.mean([q["it"]["nP"] for q in J])
        print(f"- 반복 {nm}: 프레임당 뉴턴 {nN:.1f} · PCG {nP:.1f}")
