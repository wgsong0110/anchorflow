#!/usr/bin/env python3
"""절 CM194: 한 판 요약 -> <D>/gq.json (summary.json 의 완주·오차·시간 + 모래시계 에너지 비 hgm.jsonl + 벌점 검사·훅 기록 + 물리·시각·이방성 json 이 있으면 합침).
사용: gqj_post.py <D> <장면> <판> <조건> <모드> <반복> <rc>"""
import json, os, sys
D, scene, var, cond, mode, rep, rc = sys.argv[1:8]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
M = dict(scene=scene, var=var, cond=cond, mode=mode, rep=int(rep), rc=int(rc))
try:
    s = json.load(open(f"{D}/summary.json"))
    M.update(frames=s["frames"], frames_done=s["frames_done"], diverged=s.get("diverged"), err=s.get("err_mean_done"), ms=s.get("ms_mean"), ms_min=s.get("ms_min"),
             ms_med=s.get("ms_median"))
except Exception as e:
    M["no_summary"] = repr(e)[:200]
try:
    from anchorflow import fem_gq1 as GQ
    if os.path.exists(f"{D}/hgm.jsonl"):
        h = GQ.summarize(f"{D}/hgm.jsonl"); M["hgm"] = {k: h[k] for k in ("n", "ratio", "frame_max", "frame_mean", "nbad")}; M["hgm_frames"] = h["frames"]
except Exception as e:
    M["hgm_err"] = repr(e)[:200]
for k, fn in (("gq1info", "gq1info.json"), ("gq1ck", "gq1ck.json"), ("phys", "phys.json"), ("vis", "vis.json"), ("aniso", "aniso.json")):
    try:
        o = json.load(open(f"{D}/{fn}"))
        if k == "gq1info":
            o = {q: o.get(q) for q in ("cfg", "mods", "gauss_patch", "sgauss_patch", "hooks", "coarse_swap", "err", "hm_err", "check", "bg_is_center", "vs_cfg")}
        if k == "aniso":
            o = {q: o.get(q) for q in ("max", "p99", "rel_max", "rel_p99", "rest_max", "rest_p99", "nbad")}
        M[k] = o
    except Exception:
        pass
try:                                                                                 # 발산 프레임: 로그의 마지막 프레임 줄
    L = [l for l in open(f"{D}/log.txt", errors="replace") if l.lstrip().startswith("프레임 ")]
    M["last_frame_line"] = L[-1].strip()[:160] if L else None
except Exception:
    pass
json.dump(M, open(f"{D}/gq.json", "w"), indent=1, ensure_ascii=False)
print("[gq post]", json.dumps({k: M.get(k) for k in ("frames_done", "frames", "err", "ms", "diverged")}), "hg", (M.get("hgm") or {}).get("ratio"))
