"""종단 검사: 학습·검증·지표·재개·--out_var 를 실제로 태워 본다.

지금까지 나온 버그들이 전부 "경로가 실행되지 않거나 전역이 오염돼 조용히 틀리는"
부류였다. 눈으로 읽어서는 계속 놓치므로 **짧게라도 실제로 돌려** 확인한다.

  1) 풀 학습 + 검증 + 지표가 끝까지 도는가 (진단 두 종이 찍히는가)
  2) 재개가 되는가 (체크포인트를 읽고 스텝이 이어지는가)
  3) --out_var 가 검증과 함께 돌아가는가 (훅이 평가에서 끊기는가)
  4) 풀 상태가 체크포인트에 실려 재개되는가
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--work", default=os.environ.get("AF_WORK", "/home/dkta/work"))
ap.add_argument("--gpu", default="0")
a = ap.parse_args()
W = a.work
OK = [0, 0]


def chk(name, cond, info=""):
    OK[1] += 1
    OK[0] += bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  {info}" if info else ""))


env = dict(os.environ, AF_WORK=W, CUDA_VISIBLE_DEVICES=a.gpu,
           PYTHONPATH=os.path.join(HERE, "..", "lib"),
           MPLCONFIGDIR="/home/dkta/.mplcache", PYTHONIOENCODING="utf-8")


def run(tag, extra, iters=40):
    cmd = [sys.executable, "-u", os.path.join(HERE, "train_deform.py"),
           "--data", f"{W}/one_traj_h2", "--out", f"{W}/e2e", "--tag", tag,
           "--no_mat", "--control", "--n_ctrl", "2",
           "--arch", "sgnn", "--n_nodes", "32", "--gnn_layers", "1",
           "--hidden", "128", "--obj", "pts", "--dt_cond", "--dt_scale",
           "--v_from_dt", "--det_eps", "0.1", "--det_w", "100",
           "--det_every", "20", "--lambda_bc", "1.0",
           "--lr", "3e-4", "--batch", "2", "--seed", "3",
           "--pool_fill", f"{W}/poolfill", "--pool_combos", "mic_clayC",
           "--phase2", "--phys_w", "1.0", "--iters", str(iters),
           "--save_every", "20"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       stdin=subprocess.DEVNULL)
    return (r.stdout or "") + (r.stderr or ""), r.returncode


# 1) 학습 + 검증 + 지표
o1, rc1 = run("E2E", ["--val_every", "20", "--val_n", "1", "--val_len", "5",
                      "--metrics", "--eval_t0", "3", "--eval_len", "4"])
chk("학습+검증+지표가 끝까지 돈다", rc1 == 0 and "DEFORM_OK" in o1,
    f"rc={rc1}" + ("" if rc1 == 0 else "  " + o1[-400:]))
chk("검증이 찍힌다", "[검증 " in o1, "")
chk("det 진단이 찍힌다", "[det " in o1, "")
chk("구속 진단이 찍힌다", "[구속 " in o1, "")
chk("지표(CD/EMD)가 나온다", "[지표]" in o1, "")
_pen = re.findall(r"바닥아래 ([0-9.e+-]+)", o1)
chk("바닥 관통이 0 이다", bool(_pen) and max(float(q) for q in _pen) == 0.0,
    f"최대 {max(_pen, key=float) if _pen else '?'}")
# 하드 손잡이면 w=1 인 입자의 변위가 명령과 정확히 같아야 한다
_ce = re.findall(r"손잡이오차 ([0-9.e+-]+)", o1)
chk("손잡이 변위가 명령과 일치한다",
    bool(_ce) and max(float(q) for q in _ce) < 1e-6,
    f"최대 {max(_ce, key=float) if _ce else '?'}")

# 2) 재개
ck = f"{W}/e2e/E2E_last.pt"
chk("체크포인트가 저장됐다", os.path.exists(ck), ck)
if os.path.exists(ck):
    o2, rc2 = run("E2E", ["--resume", ck, "--val_every", "100000"], iters=60)
    chk("재개가 된다", rc2 == 0 and "[재개]" in o2,
        f"rc={rc2}" + ("" if rc2 == 0 else "  " + o2[-400:]))
    chk("풀 상태가 재개된다", "[재개] 풀" in o2,
        re.search(r"\[재개\] 풀 [^\n]*", o2).group(0)[:60]
        if "[재개] 풀" in o2 else "")

# 3) out_var + 검증 (훅이 평가에서 끊기는가)
o3, rc3 = run("E2EOV", ["--out_var", "--out_var_lr", "3e-3",
                        "--val_every", "20", "--val_n", "1", "--val_len", "4"])
chk("--out_var 가 검증과 함께 돈다", rc3 == 0 and "DEFORM_OK" in o3,
    f"rc={rc3}" + ("" if rc3 == 0 else "  " + o3[-500:]))
chk("--out_var 에서 검증이 찍힌다", "[검증 " in o3, "")
print(f"\n{OK[0]}/{OK[1]}  " + ("ALL-OK" if OK[0] == OK[1] else "SOME-FAIL"))
