"""운동량 잔차 변위 망 학습·평가 (사용자 10-27 확정 설계, doc/mom_lego.md, lib/anchorflow/mom_core.py · mom_data.py).

  --mode cover  생성 분포 vs 실제 롤아웃 상태 (정답 궤적 + 기존 FEM 롤아웃 덤프) 분포 비교 + 노드 힘 검산 -> cover.json, cover.txt
  --mode train  자기지도 학습 (무작위 상태, 정답 궤적 무관). 반복마다 --batch 개 상태의 손실 평균으로 Adam 한 걸음.
                --eval_every 마다 80 프레임 롤아웃 (h = 1/60) 오차를 TB 에 (검증 지평 = 평가 지평), 최저 오차 체크포인트 ck_best.pt.
                --ckpt_every 마다 ck.pt (망·옵티마이저·반복·모든 난수 상태·평가 기록) -> --r2 로 바로 복사. 다시 띄우면 ck.pt 에서 이어 돈다.
  --mode roll   --resume 체크포인트로 롤아웃 -> roll_<name>.pt (AF_ROLL_DUMP 형식: lb_roll_to_h5·lb_render 그대로), summary json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from tqdm import tqdm                                            # noqa: E402

from anchorflow import deform_model as DM                        # noqa: E402
from anchorflow import mom_core as MC                            # noqa: E402
from anchorflow import mom_data as MD                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["cover", "train", "roll"], required=True)
ap.add_argument("--pt", required=True, help="lego L000 .pt (one/L000/L000.pt)")
ap.add_argument("--node_h", type=float, default=0.028284)
ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=30000)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--clip", type=float, default=1.0, help="기울기 전체 노름 자르기 (0 이면 끔)")
ap.add_argument("--eval_every", type=int, default=1000)
ap.add_argument("--eval_len", type=int, default=80)
ap.add_argument("--ckpt_every", type=int, default=500)
ap.add_argument("--log_every", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--k_inv", type=float, default=1.0, help="뒤집힘 장벽 κ (기존 --inv_barrier 와 같은 뜻: 에너지밀도 κ μ b(det))")
ap.add_argument("--jhat", type=float, default=0.3)
ap.add_argument("--k_bc", type=float, default=1.0, help="바닥 장벽 κ (기존 --bc_kappa 와 같은 뜻: κ m/h² b(d))")
ap.add_argument("--n_stats", type=int, default=64, help="입력 표준화 통계를 잴 생성 상태 수 (새로 시작할 때 한 번)")
ap.add_argument("--n_cover", type=int, default=256)
ap.add_argument("--fem_dump", default="", help="--mode cover: 기존 FEM 롤아웃 덤프 (roll_fem.pt, pred·F·v)")
ap.add_argument("--resume", default="", help="--mode roll: 체크포인트; --mode train: 기본 <out>/ck.pt")
ap.add_argument("--r2", default="", help="체크포인트를 저장 즉시 복사할 rclone 경로 (디렉토리)")
ap.add_argument("--name", default="mom")
ap.add_argument("--tag", default="L000")
ap.add_argument("--commit", default="")
ap.add_argument("--div_ext", type=float, default=20.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
T0 = time.time()
sc = MC.Scene(a.pt, a.node_h, dev, k_inv=a.k_inv, jhat=a.jhat, k_bc=a.k_bc)
stress = MC.FcrStress(sc.cfg)
print(f"[장면] {sc.describe()}", flush=True)
print(f"[설계] 입력 {MC.N_FEAT} = {MC.FEAT_NAMES}; 응력 모델 {stress.name}; 손실 = (Σ m/(2h²)|Δx − x̃|² + Σ V κ_inv μ b(det A; Ĵ {a.jhat}) "
      f"+ Σ κ_bc m/h² b(d; d̂ {sc.dhat:.5f})) / E_ref {sc.E_ref:.6g} (κ_inv {a.k_inv}, κ_bc {a.k_bc})", flush=True)


def sync():
    torch.cuda.synchronize()


def bg_copy(path, rel):
    if a.r2:
        subprocess.Popen(["rclone", "copyto", path, f"{a.r2.rstrip('/')}/{rel}", "-q"], stdin=subprocess.DEVNULL)


# ---------------------------------------------------------------- 롤아웃 (평가·덤프 공통)
@torch.no_grad()
def rollout(net, L, keep=False, desc="롤아웃"):
    x, v, F = sc.x0.clone(), sc.v0.clone(), sc.F0.clone()
    h = sc.frame_dt
    errs, recs, P_, F_, V_ = [], [], [], [], []
    div_at = None
    for i in tqdm(range(L), desc=desc, ncols=100, mininterval=5, leave=False):
        sync(); t_ = time.time()
        if div_at is None:
            x2, v2, F2, st = MC.step(net, sc, x, v, F, h, stress)
            sync(); dtf = time.time() - t_
            fin = bool(torch.isfinite(x2).all() and torch.isfinite(F2).all())
            mv = float((x2 - sc.x0).norm(dim=-1).max()) if fin else math.inf
            if not fin or mv > a.div_ext * sc.EXT:
                div_at = i + 1                                       # 발산: 그 앞 상태에 멈춰 두고 남은 프레임도 같은 지표로 (평균은 80 프레임 그대로)
                st = dict(st, diverged=True)
            else:
                x, v, F = x2, v2, F2
        else:
            st, dtf = dict(diverged=True, nodes=0, tets=0, inv_tets=0, detA_min=float("nan")), float("nan")
        detF = torch.linalg.det(F)
        e = DM.frame_err(x, sc.gt(i + 1), sc.EXT)
        errs.append(e)
        recs.append(dict(frame=i + 1, err=e, sec=dtf, inv_particles=int((detF <= 0).sum()), detF_min=float(detF.min()), **st))
        if keep:
            P_.append(x.cpu()); F_.append(F.cpu()); V_.append(v.cpu())
    ts = [r["sec"] for r in recs[1:] if r["sec"] == r["sec"]]
    summ = dict(err_mean=float(np.mean(errs)), err_last=errs[-1], diverged_at=div_at, inv_tets_sum=sum(r["inv_tets"] for r in recs),
                inv_tets_max=max(r["inv_tets"] for r in recs), inv_particles_max=max(r["inv_particles"] for r in recs),
                sec_per_frame=float(np.mean(ts)) if ts else float("nan"), sec_per_frame_med=float(np.median(ts)) if ts else float("nan"))
    return summ, recs, (P_, F_, V_)


# ---------------------------------------------------------------- cover
if a.mode == "cover":
    gen = torch.Generator(device=dev).manual_seed(a.seed + 999)
    sg = MD.StateGen(sc, dev)
    src = {"정답 L000 (h=1/60)": [], "생성": []}
    # 노드 힘 검산 (0 프레임 + 생성 하나)
    P0 = MC.prepare(sc, sc.x0, sc.v0, sc.F0, sc.frame_dt, stress)
    print(f"[노드 힘 검산] 정답 0 프레임: f vs −∂ΣVψ/∂Δx 상대차 {MC.check_force(sc, P0, stress):.3e} (F0 가 항등이면 힘 0 근처)", flush=True)
    for t in tqdm(range(0, sc.n_frames), desc="정답 상태", ncols=100):
        x = sc.gt(t)
        v = DM.take(sc.d["v"][t], sc.idx, dev).float()
        F = DM.traj_F(sc.d, sc.frame_dt, dev)[t].to(dev).float().reshape(-1, 3, 3)
        P = MC.prepare(sc, x, v, F, sc.frame_dt, stress)
        src["정답 L000 (h=1/60)"].append(MD.state_stats(sc, P, x, F))
        if t == 40:
            print(f"[노드 힘 검산] 정답 40 프레임: 상대차 {MC.check_force(sc, P, stress):.3e}", flush=True)
    if a.fem_dump and os.path.exists(a.fem_dump):
        D = torch.load(a.fem_dump, map_location="cpu", weights_only=False)
        src["FEM 롤아웃 (h=1/60)"] = []
        for t in tqdm(range(D["pred"].shape[0]), desc="FEM 상태", ncols=100):
            x = D["pred"][t].to(dev).float(); F = D["F"][t].to(dev).float().reshape(-1, 3, 3)
            if D.get("v") is not None:
                v = D["v"][t].to(dev).float()
            else:                                                    # 덤프에 속도가 없으면 위치 차분 (첫 프레임은 x0 에서)
                xp_ = D["pred"][t - 1].to(dev).float() if t > 0 else D["x0"].to(dev).float()
                v = (x - xp_) / sc.frame_dt
            P = MC.prepare(sc, x, v, F, sc.frame_dt, stress)
            src["FEM 롤아웃 (h=1/60)"].append(MD.state_stats(sc, P, x, F))
    infos = []
    for k in tqdm(range(a.n_cover), desc="생성 상태", ncols=100):
        x, v, F, h, info = sg.sample(gen)
        P = MC.prepare(sc, x, v, F, h, stress)
        src["생성"].append(MD.state_stats(sc, P, x, F))
        infos.append(info)
        if k == 0:
            print(f"[노드 힘 검산] 생성 0: 상대차 {MC.check_force(sc, P, stress):.3e} | 상태 {info}", flush=True)
    qs = [0.0, 0.01, 0.5, 0.99, 1.0]
    res = {}
    keys = list(src["생성"][0].keys())
    lines = [f"[분포 비교] 분위 {qs} (사면체·노드·입자 값은 상태마다 최대 2 만 개 부분표본을 모아서)"]
    for kname in keys:
        lines.append(f"  {kname}")
        res[kname] = {}
        for s, L_ in src.items():
            if not L_:
                continue
            t = torch.cat([d[kname] for d in L_])
            q = torch.quantile(t.double(), torch.tensor(qs, dtype=torch.float64)).tolist() if t.numel() > 1 else [float(t)] * len(qs)
            res[kname][s] = q
            lines.append(f"    {s:<22} " + " ".join(f"{z:>11.4g}" for z in q))
    # 덮는지: 실제 분포의 [1%, 99%] 가 생성 [min, max] 안
    lines.append("[덮음 판정] 실제 [1%, 99%] ⊂ 생성 [최소, 최대]")
    cov = {}
    for kname in keys:
        g_ = res[kname]["생성"]
        for s in src:
            if s == "생성" or s not in res[kname]:
                continue
            r_ = res[kname][s]
            ok = (r_[1] >= g_[0] - 1e-12) and (r_[3] <= g_[4] + 1e-12)
            cov[f"{kname} | {s}"] = ok
            lines.append(f"  {'O' if ok else 'X'} {kname} | {s}: 실제 [{r_[1]:.4g}, {r_[3]:.4g}] 생성 [{g_[0]:.4g}, {g_[4]:.4g}]")
    hs = [i["h"] for i in infos]
    lines.append(f"[생성 dt] 1/60 비율 {np.mean([abs(z - sc.frame_dt) < 1e-12 for z in hs]):.3f}, 범위 {min(hs):.3e}~{max(hs):.3e}")
    txt = "\n".join(lines)
    print(txt, flush=True)
    open(os.path.join(a.out, "cover.txt"), "w").write(txt + "\n")
    json.dump(dict(quantiles=qs, res=res, cover=cov, infos=infos), open(os.path.join(a.out, "cover.json"), "w"), ensure_ascii=False, indent=1)
    print("MOM_COVER_DONE", flush=True)
    sys.exit(0)

# ---------------------------------------------------------------- 망
net = MC.build_net(sc, dev=dev)
print(f"[망] Graph U-Net 3 수준 (gunet5_near 몸통), 파라미터 {sum(p.numel() for p in net.parameters())}", flush=True)

if a.mode == "roll":
    ck = torch.load(a.resume, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["net"]); net.eval()
    print(f"[롤아웃] 체크포인트 {a.resume} (반복 {ck['it']})", flush=True)
    summ, recs, (P_, F_, V_) = rollout(net, a.eval_len, keep=True, desc=f"롤아웃 {a.tag}")
    G_ = [sc.gt(i + 1).cpu() for i in range(a.eval_len)]
    dp = os.path.join(a.out, f"roll_{a.name}.pt")
    torch.save({"pred": torch.stack(P_), "gt": torch.stack(G_), "x0": sc.x0.cpu(), "ctrl_pos": None, "ctrl_R": None, "ctrl_id": None,
                "t0": 0, "tag": a.tag, "nodes": None, "fscal": None, "gt_fscal": None, "EXT": sc.EXT,
                "F": torch.stack(F_).float(), "v": torch.stack(V_).float(),
                "mom": dict(ckpt=a.resume, it=ck["it"], node_h=a.node_h, commit=a.commit, **summ)}, dp)
    with open(os.path.join(a.out, f"stats_{a.name}.jsonl"), "w") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(dict(summ, it=ck["it"], ckpt=a.resume, commit=a.commit), open(os.path.join(a.out, f"summary_{a.name}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[롤아웃] {a.tag} t0=  0: {a.eval_len} 프레임, 평균 {100 * summ['err_mean']:.3f}% | 뒤집힌 사면체 합 {summ['inv_tets_sum']} 최대 {summ['inv_tets_max']} "
          f"| 뒤집힌 입자 최대 {summ['inv_particles_max']} | 프레임당 {summ['sec_per_frame']:.4f} s (중앙 {summ['sec_per_frame_med']:.4f}) "
          f"| 발산 {summ['diverged_at']} -> {dp}", flush=True)
    bg_copy(dp, os.path.basename(dp))
    print("MOM_ROLL_DONE", flush=True)
    sys.exit(0)

# ---------------------------------------------------------------- train
from torch.utils.tensorboard import SummaryWriter              # noqa: E402

opt = torch.optim.Adam(net.parameters(), lr=a.lr)
gen = torch.Generator(device=dev)
sg = MD.StateGen(sc, dev)
ckp = a.resume or os.path.join(a.out, "ck.pt")
it0, best, hist = 0, None, []
if os.path.exists(ckp):
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
    it0, best, hist = int(ck["it"]), ck.get("best"), ck.get("hist", [])
    gen.set_state(ck["rng"]["gen"])
    torch.set_rng_state(ck["rng"]["cpu"]); torch.cuda.set_rng_state(ck["rng"]["cuda"])
    np.random.set_state(ck["rng"]["np"]); random.setstate(ck["rng"]["py"])
    print(f"[이어 돈다] {ckp} 반복 {it0} 부터, 최저 {best}", flush=True)
else:
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    gen.manual_seed(a.seed + 12345)
    with torch.no_grad():
        fs = []
        for _ in tqdm(range(a.n_stats), desc="입력 통계", ncols=100):
            x, v, F, h, _info = sg.sample(gen)
            fs.append(MC.prepare(sc, x, v, F, h, stress).feat)
        net.set_input_stats(torch.cat(fs))
    print(f"[입력 표준화] 평균 {[round(z, 4) for z in net.in_mu.tolist()]} 표준편차 {[round(z, 4) for z in net.in_sd.tolist()]}", flush=True)


def save_ck(path, it):
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), it=it, best=best, hist=hist, args=vars(a), n_feat=MC.N_FEAT,
                    feat_names=MC.FEAT_NAMES, commit=a.commit,
                    rng=dict(gen=gen.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(), np=np.random.get_state(),
                             py=random.getstate())), path + ".tmp")
    os.replace(path + ".tmp", path)
    bg_copy(path, os.path.basename(path))


tb = SummaryWriter(os.path.join(a.out, "tb"))
evl = open(os.path.join(a.out, "evals.jsonl"), "a")
pbar = tqdm(range(it0, a.iters), desc="운동량 잔차 학습", ncols=110, mininterval=10, initial=it0, total=a.iters)
acc, t_last = {}, time.time()
for it in pbar:
    net.train()
    opt.zero_grad(set_to_none=True)
    rec = dict(res=0.0, inv=0.0, bc=0.0, loss=0.0, inv_net=0.0, inv_exp=0.0, any_exp=0.0, pen=0.0, h=0.0)
    for b in range(a.batch):
        with torch.no_grad():
            x, v, F, h, info = sg.sample(gen)
            P = MC.prepare(sc, x, v, F, h, stress)
            Jx = torch.linalg.det(MC.cell_A(P.S, MC.explicit_dx(sc, P)))
        dx = MC.net_dx(net, sc, P.S, P.feat, h)
        terms, J, sds = MC.loss_terms(sc, P, dx, x)
        loss = (terms["res"] + terms["inv"] + terms["bc"]) / a.batch
        loss.backward()
        rec["res"] += float(terms["res"]) / a.batch; rec["inv"] += float(terms["inv"]) / a.batch; rec["bc"] += float(terms["bc"]) / a.batch
        rec["loss"] += float(loss)
        rec["inv_net"] += float((J <= 0).float().mean()) / a.batch
        rec["inv_exp"] += float((Jx <= 0).float().mean()) / a.batch
        rec["any_exp"] += float((Jx <= 0).any()) / a.batch
        rec["pen"] += (float((sds[0] < 0).float().mean()) if sds else 0.0) / a.batch
    if not all(math.isfinite(v_) for v_ in rec.values()):
        print(f"[경고] 반복 {it + 1} 손실 비유한 {rec} -- 이 걸음 건너뜀", flush=True)
        opt.zero_grad(set_to_none=True)
    else:
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip if a.clip > 0 else float("inf"))
        rec["gnorm"] = float(gn)
        opt.step()
    for k, v_ in rec.items():
        acc.setdefault(k, []).append(v_)
    k1 = it + 1
    if k1 % a.log_every == 0:
        m_ = {k: float(np.mean(v_)) for k, v_ in acc.items()}
        acc = {}
        tb.add_scalar("학습/손실_합", m_["loss"], k1)
        tb.add_scalar("학습/잔차_항", m_["res"], k1)
        tb.add_scalar("학습/뒤집힘_장벽_항", m_["inv"], k1)
        tb.add_scalar("학습/바닥_장벽_항", m_["bc"], k1)
        tb.add_scalar("학습/망_뒤집힌_사면체_비율", m_["inv_net"], k1)
        tb.add_scalar("학습/명시해_뒤집힌_사면체_비율", m_["inv_exp"], k1)
        tb.add_scalar("학습/명시해_뒤집힘_있는_표본_비율", m_["any_exp"], k1)
        tb.add_scalar("학습/망_바닥_관통_입자_비율", m_["pen"], k1)
        if "gnorm" in m_:
            tb.add_scalar("학습/기울기_노름", m_["gnorm"], k1)
        tb.add_scalar("학습/초당_반복", a.log_every / max(time.time() - t_last, 1e-9), k1)
        t_last = time.time()
        pbar.set_postfix_str(f"손실 {m_['loss']:.3e} 잔차 {m_['res']:.3e} 뒤집힘 {m_['inv']:.2e} 바닥 {m_['bc']:.2e} 망뒤집힘 {m_['inv_net']:.1e}")
        if k1 % (a.log_every * 10) == 0:
            print(f"  [반복 {k1}] 손실 {m_['loss']:.4e} (잔차 {m_['res']:.4e} 뒤집힘 {m_['inv']:.3e} 바닥 {m_['bc']:.3e}) | 망 뒤집힌 사면체 {m_['inv_net']:.2e} "
                  f"명시해 {m_['inv_exp']:.2e} (있는 표본 {m_['any_exp']:.2f}) | 관통 입자 {m_['pen']:.2e} | {time.time() - T0:.0f} s", flush=True)
    if k1 % a.eval_every == 0 or k1 == a.iters:
        net.eval()
        summ, recs, _ = rollout(net, a.eval_len, desc=f"평가 {k1}")
        tb.add_scalar("롤아웃_80/위치오차_평균_퍼센트", 100 * summ["err_mean"], k1)
        tb.add_scalar("롤아웃_80/위치오차_마지막_퍼센트", 100 * summ["err_last"], k1)
        tb.add_scalar("롤아웃_80/뒤집힌_사면체_합", summ["inv_tets_sum"], k1)
        tb.add_scalar("롤아웃_80/뒤집힌_사면체_최대", summ["inv_tets_max"], k1)
        tb.add_scalar("롤아웃_80/뒤집힌_입자_최대", summ["inv_particles_max"], k1)
        tb.add_scalar("롤아웃_80/프레임당_시간_초", summ["sec_per_frame"], k1)
        tb.add_scalar("롤아웃_80/발산_프레임", summ["diverged_at"] or 0, k1)
        hist.append(dict(it=k1, **summ))
        evl.write(json.dumps(dict(it=k1, **summ), ensure_ascii=False) + "\n"); evl.flush()
        isbest = best is None or summ["err_mean"] < best["err_mean"]
        if isbest:
            best = dict(it=k1, **summ)
        print(f"[평가] 반복 {k1}: 80 프레임 롤아웃 위치 오차 평균 {100 * summ['err_mean']:.3f}% (마지막 {100 * summ['err_last']:.3f}%) | "
              f"뒤집힌 사면체 합 {summ['inv_tets_sum']} 최대 {summ['inv_tets_max']} | 뒤집힌 입자 최대 {summ['inv_particles_max']} | "
              f"프레임당 {summ['sec_per_frame']:.4f} s | 발산 {summ['diverged_at']} | 최저 {100 * best['err_mean']:.3f}% (반복 {best['it']})", flush=True)
        tb.flush()
        if isbest:
            save_ck(os.path.join(a.out, "ck_best.pt"), k1)
    if k1 % a.ckpt_every == 0 or k1 == a.iters:
        save_ck(os.path.join(a.out, "ck.pt"), k1)
tb.close(); evl.close()
print(f"[학습 끝] 반복 {a.iters}, 최저 {best}, {time.time() - T0:.0f} s", flush=True)
print("MOM_TRAIN_DONE", flush=True)
