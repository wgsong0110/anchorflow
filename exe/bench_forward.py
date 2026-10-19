"""예전 앵커 attention 모델(41300e3 NextStep)의 forward 시간을 잰다.

사용:
  python exe/bench_forward.py old --wt <41300e3 worktree> --out <dir>
      NextStep(은닉 128, attention 4층, head 4, 앵커 512) 을 무작위 입력으로
      배치 64 / 배치 1 forward 시간 (no_grad 와 grad 둘 다), 그리고
      렌더용 K=8 shape-matching 스키닝(가우시안 20만) 시간을 따로 잰다.
      스키닝은 학습 경로에 없었다 (렌더 때만).
  python exe/bench_forward.py parse --log <train_deform 로그> --tag <이름> --out <dir>
      AF_PROF=1 롤아웃 로그의 [롤아웃 시간]/[프레임] 줄을 results.json 에 넣는다.
결과는 <out>/results.json 에 키별로 합쳐 쓴다. 시간은 synchronize 로 감싸 여러 번 잰 중앙값(ms).
"""
import argparse
import json
import os
import re
import statistics
import sys
import time


def _merge(out, key, val):
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, "results.json")
    d = json.load(open(fn)) if os.path.exists(fn) else {}
    d[key] = val
    json.dump(d, open(fn, "w"), ensure_ascii=False, indent=1)


def _time(fn, warm=10, reps=50):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(ts), "min_ms": min(ts), "max_ms": max(ts), "reps": reps}


def old(args):
    import torch
    sys.path.insert(0, os.path.join(args.wt, "lib"))
    from anchorflow import nextstate as NS
    dev = "cuda"
    torch.manual_seed(0)
    net = NS.NextStep(hidden=128, depth=4, heads=4).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    print(f"[예전 모델] NextStep 파라미터 {npar:,} 개 (기대 약 73만), fused geobias 사용 가능={NS._HAVE_GEOBIAS}", flush=True)
    res = {"params": npar, "fused_geobias": bool(NS._HAVE_GEOBIAS), "gpu": torch.cuda.get_device_name(0),
           "contended": bool(os.environ.get("BENCH_CONTENDED")), "M": 512}
    M = 512
    dt = torch.tensor(1.0 / 30.0, device=dev)
    for B in (64, 1):
        p = torch.rand(B, M, 3, device=dev)
        v = torch.randn(B, M, 3, device=dev) * 1e-2
        a = torch.randn(B, M, 3, device=dev) * 1e-1

        def f_ng():
            with torch.no_grad():
                net(p, v, a, dt)

        def f_g():      # 학습 경로: grad 켜짐 → eager 기하 편향
            out = net(p, v, a, dt)
            return out

        def f_gb():     # 학습 1반복의 forward+backward
            net.zero_grad(set_to_none=True)
            net(p, v, a, dt).square().mean().backward()
        for nm, fn in (("forward_nograd", f_ng), ("forward_grad", f_g), ("forward_backward", f_gb)):
            try:
                r = _time(fn)
            except torch.cuda.OutOfMemoryError as e:
                r = {"error": f"OOM {e}"[:200]}
                torch.cuda.empty_cache()
            res[f"B{B}_{nm}"] = r
            print(f"[예전 모델] 배치 {B} {nm}: {r}", flush=True)
            _merge(args.out, "old_nextstep", res)

    # 렌더용 K=8 shape-matching 스키닝 (학습 경로에는 없었다)
    N, K = 200_000, 8
    a0 = torch.rand(M, 3, device=dev)
    g0 = torch.rand(N, 3, device=dev)
    d2 = torch.cdist(g0, a0)
    w, gi = (-d2).topk(K, dim=1)
    w = torch.softmax(w / 0.02, 1)                                  # [N,K]
    an = torch.cdist(a0, a0).topk(K + 1, largest=False).indices     # [M,K+1] 앵커 이웃(자기 포함)
    a1 = a0 + torch.randn_like(a0) * 1e-2

    def skin():
        with torch.no_grad():
            P0 = a0[an] - a0[:, None]                               # [M,K+1,3]
            P1 = a1[an] - a1[:, None]
            A = P1.transpose(1, 2) @ P0                             # [M,3,3]
            U, S, Vh = torch.linalg.svd(A)
            dd = torch.det(U @ Vh)
            U = torch.cat([U[..., :2], U[..., 2:] * dd[:, None, None]], -1)
            R = U @ Vh                                              # 극분해 회전
            loc = g0[:, None] - a0[gi]                              # [N,K,3]
            x = torch.einsum("nkij,nkj->nki", R[gi], loc) + a1[gi]
            return (w[..., None] * x).sum(1)
    r = _time(skin)
    r.update({"N_gauss": N, "K": K, "note": "렌더 전용, 학습 경로에 없음"})
    res["skin_K8_200k"] = r
    print(f"[예전 모델] K=8 shape-matching 스키닝 (가우시안 20만, 렌더 전용): {r}", flush=True)
    _merge(args.out, "old_nextstep", res)


def parse(args):
    txt = open(args.log, encoding="utf-8", errors="replace").read()
    r = {"log": args.log, "contended": bool(os.environ.get("BENCH_CONTENDED"))}
    m = re.findall(r"\[롤아웃 시간\][^\n]*", txt)
    r["rollout_line"] = m[-1] if m else None
    if m:
        mm = re.search(r"(\d+) 프레임 ([\d.]+) s \(프레임당 ([\d.]+) s\) \|(.*)", m[-1])
        if mm:
            r["frames"] = int(mm.group(1)); r["total_s"] = float(mm.group(2))
            r["per_frame_s"] = float(mm.group(3))
            r["sections_s"] = {k: float(v) for k, v in re.findall(r"(\S+) ([\d.]+)s", mm.group(4))}
    r["frame_lines"] = re.findall(r"\[프레임\][^\n]*", txt)[-3:]
    r["count_lines"] = [l for l in txt.splitlines() if re.search(r"노드|입자|파라미터|n_nodes|n_part", l)][:20]
    _merge(args.out, args.tag, r)
    print(f"[{args.tag}] {r.get('rollout_line')}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["old", "parse"])
    ap.add_argument("--wt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log")
    ap.add_argument("--tag")
    a = ap.parse_args()
    old(a) if a.mode == "old" else parse(a)
