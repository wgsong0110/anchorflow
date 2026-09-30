"""학습 손실에 **모든 항이 실제로 들어가는지** 검사한다.

교사 경로를 지울 때 det 마스킹·복구 벌점과 L_bc 가 손실에서 통째로 빠졌는데
진단 출력도 같은 continue 뒤에 있어 몇 시간을 모르고 학습했다. 같은 일을 막으려
**손실이 각 인자에 실제로 반응하는지** 를 스크립트로 확인한다.

방법: 같은 씨앗으로 아주 짧게 돌려 첫 스텝의 손실을 읽고, 인자를 바꿨을 때
값이 달라지는지 본다. 달라지지 않으면 그 인자는 손실에 안 들어간 것이다.
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--work", default=os.environ.get("AF_WORK", "/home/dkta/work"))
ap.add_argument("--gpu", default="0")
ap.add_argument("--iters", type=int, default=12)
a = ap.parse_args()
W = a.work
OK = [0, 0]


def chk(name, cond, info=""):
    OK[1] += 1
    OK[0] += bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  {info}" if info else ""))


BASE = [
    sys.executable, "-u", os.path.join(HERE, "train_deform.py"),
    "--data", f"{W}/one_traj_h2", "--out", f"{W}/lt", "--tag", "LT",
    "--no_mat", "--control", "--n_ctrl", "2",
    "--arch", "sgnn", "--n_nodes", "32", "--gnn_layers", "1", "--hidden", "128",
    "--obj", "pts", "--dt_cond", "--dt_scale", "--v_from_dt",
    "--lr", "0", "--batch", "2", "--seed", "7",
    "--pool_fill", f"{W}/poolfill", "--pool_combos", "mic_clayC",
    "--pool_fresh", "1.0",               # 항상 정지 상태에서 시작 -> 결정적
    "--phase2", "--iters", str(a.iters),
    "--val_every", "1000000", "--save_every", "1000000", "--det_every", "4",
]
env = dict(os.environ, AF_WORK=W, CUDA_VISIBLE_DEVICES=a.gpu,
           PYTHONPATH=os.path.join(HERE, "..", "lib"),
           MPLCONFIGDIR="/home/dkta/.mplcache", PYTHONIOENCODING="utf-8")


def run(extra):
    r = subprocess.run(BASE + extra, capture_output=True, text=True, env=env,
                       stdin=subprocess.DEVNULL)
    out = (r.stdout or "") + (r.stderr or "")
    es = re.findall(r"E=([0-9.e+-]+)", out.replace("\r", "\n"))
    bc = re.findall(r"보정최대 ([0-9.e+-]+)", out.replace("\r", "\n"))
    if not es:
        print(out[-1200:])
    return (float(es[-1]) if es else None,
            float(bc[-1]) if bc else None, out)


e_ref, bc_ref, out_ref = run(["--det_eps", "0.1", "--det_w", "100",
                              "--lambda_bc", "1.0", "--phys_w", "1.0"])
chk("기준 실행이 손실을 낸다", e_ref is not None, f"E={e_ref}")
chk("구속 진단이 찍힌다 (풀 경로에서 도달한다)", bc_ref is not None,
    f"보정최대={bc_ref}")
chk("det 진단이 찍힌다", "[det " in out_ref, "")

e_bc0, _, _ = run(["--det_eps", "0.1", "--det_w", "100",
                   "--lambda_bc", "0.0", "--phys_w", "1.0"])
chk("--lambda_bc 가 손실을 바꾼다 (L_bc 가 들어간다)",
    e_bc0 is not None and abs(e_bc0 - e_ref) > 1e-12,
    f"1.0 -> {e_ref:.6e},  0.0 -> {e_bc0:.6e}" if e_bc0 else "")

# det 벌점: eps 를 크게 하면 유효 사면체까지 문턱 아래로 들어가 벌점이 커진다
e_d, _, _ = run(["--det_eps", "1.5", "--det_w", "100",
                 "--lambda_bc", "1.0", "--phys_w", "1.0"])
chk("--det_eps/--det_w 가 손실을 바꾼다 (det 벌점이 들어간다)",
    e_d is not None and abs(e_d - e_ref) > 1e-12,
    f"eps 0.1 -> {e_ref:.6e},  1.5 -> {e_d:.6e}" if e_d else "")

e_w, _, _ = run(["--det_eps", "0.1", "--det_w", "100",
                 "--lambda_bc", "1.0", "--phys_w", "3.0"])
chk("--phys_w 가 손실을 바꾼다",
    e_w is not None and abs(e_w - e_ref) > 1e-12,
    f"1.0 -> {e_ref:.6e},  3.0 -> {e_w:.6e}" if e_w else "")
print(f"\n{OK[0]}/{OK[1]}  " + ("ALL-OK" if OK[0] == OK[1] else "SOME-FAIL"))
