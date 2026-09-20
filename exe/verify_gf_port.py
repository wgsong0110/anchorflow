"""GF 원본과 이식본을 씬마다 **같은 초기상태에서** 굴려 입자별로 맞춰 본다.

씬 하나를 맞추는 것은 우연일 수 있다. 재질·전달 방식·경계·구동 조건을 갈라
놓고 전부 돌려야 "임의의 씬에서 같다" 고 말할 수 있다. 그래서 돌리는 것은
`exe/make_gf_scenes.py` 가 찍어낸 config 묶음이다.

한 씬의 절차는 셋이다.
  1. GF 로 돌려 h5 를 남긴다 (입자 순서가 고정된다)
  2. 그 0 프레임을 그대로 이식본의 초기상태로 준다
  3. 프레임마다 입자별 거리 차이를 물체 지름으로 나눈다

판정은 `차이 / 그동안 움직인 거리` 로 한다. 물체가 거의 안 움직인 프레임에서
절대 차이만 보면 아무 값이나 통과하기 때문이다.

**그리고 씬이 실제로 변형했는지 같이 잰다.** 처음 돌렸을 때 공이 바닥에 닿지도
못해 자유낙하만 했고, 재질이 뭐든 전달 방식이 뭐든 궤적이 똑같이 나왔다. 열 개
넘는 씬이 "일치" 로 찍혔지만 그것은 통과가 아니라 **아무것도 시험하지 못한 것**
이다. 그래서 입자 변위에서 **전체 평행이동을 뺀 나머지**의 크기를 같이 보고,
그게 0 에 가까우면 판정을 보류한다.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cfg_dir", required=True, help="make_gf_scenes.py 의 출력")
ap.add_argument("--gf_root", required=True)
ap.add_argument("--model", required=True, help="GF 3DGS 모델 디렉토리")
ap.add_argument("--work", required=True)
ap.add_argument("--out", default=None, help="요약 json")
ap.add_argument("--only", default=None, help="쉼표로 고른 태그만")
ap.add_argument("--flip", default="auto", choices=("auto", "on", "off"))
ap.add_argument("--grid", default="dense", choices=("dense", "sparse"))
ap.add_argument("--rm_h5", action="store_true", help="비교가 끝나면 h5 를 지운다")
ap.add_argument("--tol", type=float, default=0.05,
                help="차이/이동 이 이 값보다 작으면 일치로 본다")
a = ap.parse_args()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(a.work, exist_ok=True)
tags = sorted(os.path.splitext(os.path.basename(p))[0]
              for p in glob.glob(os.path.join(a.cfg_dir, "*.json")))
if a.only:
    keep = set(a.only.split(","))
    tags = [t for t in tags if t in keep]


def rd(p, key="x"):
    with h5py.File(p, "r") as h:
        d = np.array(h[key])
    return (d.T if d.shape[0] in (3, 9) else d).astype(np.float64)


def run(cmd, cwd=None, env=None):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


rows = []
for tag in tags:
    cfg = os.path.join(a.cfg_dir, f"{tag}.json")
    gdir = os.path.join(a.work, f"gf_{tag}")
    mdir = os.path.join(a.work, f"my_{tag}")
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = (a.gf_root + ":" + os.path.join(a.gf_root, "gaussian-splatting")
                         + ":" + env.get("PYTHONPATH", ""))
    if not glob.glob(os.path.join(gdir, "**", "sim_*.h5"), recursive=True):
        rc, log = run([sys.executable, "gs_simulation.py", "--model_path", a.model,
                       "--output_path", gdir, "--config", cfg, "--output_h5",
                       "--no_render"],   # 배경 체크포인트는 그림에만 쓴다
                      cwd=a.gf_root, env=env)
        if rc != 0:
            print(f"[{tag}] GF 실패 rc={rc}\n{log[-1200:]}", flush=True)
            rows.append(dict(tag=tag, ok=False, why="GF 실패")); continue
    fa = sorted(glob.glob(os.path.join(gdir, "**", "sim_*.h5"), recursive=True))
    if not fa:
        rows.append(dict(tag=tag, ok=False, why="GF h5 없음")); continue
    shutil.rmtree(mdir, ignore_errors=True)
    rc, log = run([sys.executable, os.path.join(HERE, "exe", "gf_mpm.py"),
                   "--config", cfg, "--h5", fa[0], "--out", mdir,
                   "--flip", a.flip, "--grid", a.grid])
    if rc != 0:
        print(f"[{tag}] 이식본 실패 rc={rc}\n{log[-1500:]}", flush=True)
        rows.append(dict(tag=tag, ok=False, why="이식본 실패")); continue
    fb = sorted(glob.glob(os.path.join(mdir, "sim_*.h5")))
    n = min(len(fa), len(fb))
    x0 = rd(fa[0])
    EXT = float(np.linalg.norm(x0.max(0) - x0.min(0)))
    worst, worst_f, mv_at = 0.0, 0, 0.0
    d0 = float(np.abs(rd(fa[0]) - rd(fb[0])).max())
    # 변형량: 마지막 프레임 변위에서 **평행이동을 뺀** 나머지의 RMS
    _dsp = rd(fa[n - 1]) - x0
    _ok0 = np.isfinite(_dsp).all(1)
    _res = _dsp[_ok0] - _dsp[_ok0].mean(0)
    deform = float(np.sqrt((_res ** 2).sum(1).mean()) / EXT)
    # 0 프레임은 정의상 같아야 한다 -- 그건 init_gap 으로 따로 본다.
    for i in range(1, n):
        xa, xb = rd(fa[i]), rd(fb[i])
        ok = np.isfinite(xa).all(1) & np.isfinite(xb).all(1)
        rel = float(np.linalg.norm(xa[ok] - xb[ok], axis=1).mean() / EXT)
        mv = float(np.linalg.norm(xa[ok] - x0[ok], axis=1).mean() / EXT)
        if rel > worst:
            worst, worst_f, mv_at = rel, i, mv
    ratio = worst / max(mv_at, 1e-12)
    vacuous = deform < 5e-3          # 사실상 자유낙하 -- 시험이 안 됐다
    good = bool(ratio < a.tol and d0 < 1e-6 and not vacuous)
    rows.append(dict(tag=tag, ok=bool(good), n=int(n), pts=int(x0.shape[0]),
                     init_gap=d0, worst=worst, worst_frame=worst_f,
                     moved=mv_at, ratio=ratio, deform=deform, vacuous=bool(vacuous),
                     sec=round(time.time() - t0, 1)))
    _v = "일치" if good else ("변형없음" if vacuous else "갈린다")
    print(f"[{tag:14s}] {_v:6s} 최악 프레임 {worst_f:3d}  "
          f"차이 {100*worst:7.4f}%  이동 {100*mv_at:7.3f}%  비 {ratio:.4f}  "
          f"변형 {100*deform:6.3f}%  0프레임차 {d0:.1e}  "
          f"입자 {x0.shape[0]}  {rows[-1]['sec']}s", flush=True)
    if a.rm_h5:
        shutil.rmtree(gdir, ignore_errors=True); shutil.rmtree(mdir, ignore_errors=True)

done = [r for r in rows if "ratio" in r]
bad = [r for r in rows if not r["ok"]]
print(f"\n[요약] {len(done)}/{len(rows)} 씬 비교 완료, 일치 "
      f"{len(done)-len([r for r in done if not r['ok']])}/{len(done)}")
for r in bad:
    why = r.get("why", "")
    if "ratio" in r:
        why = ("변형이 없어 시험이 안 됐다" if r.get("vacuous")
               else f"차이/이동 {r['ratio']:.4f}")
    print(f"  못 넘긴 씬: {r['tag']}  {why}")
if a.out:
    json.dump(rows, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"[저장] {a.out}")
print("VERIFY_DONE", flush=True)
