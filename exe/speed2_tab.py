"""절 CM36 묶음 speed2 표 (읽기만): 판 디렉토리의 summary.json·fem.jsonl·(있으면) pred.pt 에서 시간·오차·손상·발산·탄성·점 수.

사용:
  python exe/speed2_tab.py row <판 디렉토리...>            -- 판마다 한 줄 JSON (평균·최대 ms, 최대 프레임, 16.7 넘는 프레임 수, 점 처음·최대·최악 프레임 점, 오차, 손상, 발산, 탄성 최대)
  python exe/speed2_tab.py worst <판 디렉토리...>          -- 판들의 최악 프레임 ms 최대 (조건 단계용; 판 없거나 미완이면 nan)
  python exe/speed2_tab.py cmp <기준 a> <기준 b> <비교...>  -- pred.pt 프레임별 입자 위치 차 (최대·평균 |Δx| mm): a 대 b (비결정 범위), a 대 비교들
  python exe/speed2_tab.py prof <판 디렉토리>              -- --prof 프레임 중 재생 ms 최대 프레임의 커널별 ms (상위 25)
"""
import json
import math
import os
import sys


def _jl(d):
    p = os.path.join(d, "fem.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def row(d, warm=2):
    S = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else None
    L = _jl(d)
    o = dict(run=os.path.basename(d.rstrip("/")), done=S is not None, frames=len(L))
    if not L:
        return o
    ms = [(i + 1, r["ms"]) for i, r in enumerate(L) if i >= warm and r.get("ms") is not None and math.isfinite(r["ms"])]
    if ms:
        fw, mw = max(ms, key=lambda t: t[1])
        o.update(ms_mean=sum(m for _, m in ms) / len(ms), ms_max=mw, f_max=fw, n_over=sum(m > 16.7 for _, m in ms), nq_worst=L[fw - 1].get("nq"))
    nq = [r.get("nq") for r in L if r.get("nq")]
    if nq:
        o.update(nq0=nq[0], nq_max=max(nq))
    if S is not None:
        o.update(err=100 * S["err_mean"] if S.get("err_mean") is not None else None, err_done=100 * S["err_mean_done"], div=S.get("diverged"), sp2=S["args"].get("sp2", 0), cell=S["args"].get("cell"))
    en = [r.get("en") for r in L if r.get("en")]
    if en:
        o["dmg"] = 100 * en[-1]["lj0"]                                           # 마지막 프레임 logJp > 0 몫 (fs.lj_pos 와 같은 값)
        o["eel_max"] = max(e.get("Eel", float("nan")) for e in en)
    o["redo"] = sum(1 for r in L if r.get("redo"))
    return o


def cmp(a, b, others):
    import torch
    A = torch.load(os.path.join(a, "pred.pt"), map_location="cpu")["pred"].double()

    def one(d):
        B = torch.load(os.path.join(d, "pred.pt"), map_location="cpu")["pred"].double()
        n = min(len(A), len(B))
        dx = (A[:n] - B[:n]).norm(dim=-1)
        return [[round(1e3 * float(dx[i].max()), 5), round(1e3 * float(dx[i].mean()), 6)] for i in range(n)]
    o = {"ref": os.path.basename(a), os.path.basename(b): one(b)}
    for d in others:
        o[os.path.basename(d)] = one(d)
    return o


def prof(d):
    L = _jl(d)
    P = [(i + 1, r) for i, r in enumerate(L) if r.get("kernel_ms") and not r.get("redo")]
    if not P:
        return {}
    f, r = max(P, key=lambda t: t[1]["ms"])
    km = {k: v[0] for k, v in r["kernel_ms"].items()}
    tot = sum(km.values())
    top = sorted(km.items(), key=lambda t: -t[1])[:25]
    return dict(frame=f, ms=r["ms"], nq=r.get("nq"), sync_sum=tot, top=[[k, round(v, 3), round(100 * v / tot, 1), r["kernel_ms"][k][1]] for k, v in top])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "row":
        for d in sys.argv[2:]:
            print(json.dumps(row(d), ensure_ascii=False))
    elif cmd == "worst":
        w = float("nan")
        for d in sys.argv[2:]:
            r = row(d)
            if not r.get("done") or "ms_max" not in r:
                print("nan"); sys.exit(0)
            w = r["ms_max"] if math.isnan(w) else max(w, r["ms_max"])
        print(w)
    elif cmd == "cmp":
        print(json.dumps(cmp(sys.argv[2], sys.argv[3], sys.argv[4:])))
    elif cmd == "prof":
        print(json.dumps(prof(sys.argv[2]), ensure_ascii=False))
