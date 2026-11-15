#!/usr/bin/env python3
"""절 CM183 한 번 실행 분석 (인스턴스에서): <실행 디렉토리> <장면> <판> <반복> -> <디렉토리>/metrics.json
지표: 완주 (frames_done/frames, 발산 프레임), 오차 평균 (err_mean_done), 프레임 시간 ms 평균, 뒤집힘 (fem.jsonl pd.nJneg 최대·합, 입자 F det <= 0 입자 수 최대 = pred.pt),
에너지 (pred.pt 입자 위치·F 에서: 운동 + 중력 + 탄성 NH 근사 = 진단 대리 정의, 소성 장면은 F 가 탄성 몫이 아닐 수 있음) -- 첫 프레임 대비 최대 증가 / 규모,
모래시계 (반복 1 만: exe/q27_eig.py 첫 프레임 영 에너지 모드 수; q64 는 미지원)."""
import json, os, subprocess, sys
D, scene, var, rep = sys.argv[1:5]
M = dict(scene=scene, var=var, rep=int(rep))
try:
    s = json.load(open(f"{D}/summary.json"))
    M.update(frames=s["frames"], frames_done=s["frames_done"], diverged=s.get("diverged"), err=s.get("err_mean_done"), ms=s.get("ms_mean"), ms_min=s.get("ms_min"))
except Exception as e:
    M["no_summary"] = repr(e)
try:
    pd = [json.loads(l).get("pd") or {} for l in open(f"{D}/fem.jsonl")]
    M["nJneg_max"] = max([q.get("nJneg", 0) for q in pd], default=None)
    M["Jmin"] = min([q["Jmin"] for q in pd if "Jmin" in q], default=None)
    M["flip_frames"] = sum(1 for q in pd if q.get("nJneg", 0) > 0)
except Exception as e:
    M["fem_err"] = repr(e)
try:
    cmd = open(f"{D}/cmd.txt").read()
    data = cmd.split("--data ")[1].split()[0]
    if os.path.exists(f"{D}/pred.pt"):
        import torch
        P = torch.load(f"{D}/pred.pt", map_location="cpu", weights_only=False)
        X = P["pred"].double(); F = P.get("F")
        M["pred_frames"] = int(X.shape[0])
        if F is not None:
            F = F.double().reshape(F.shape[0], F.shape[1], 3, 3)
            det = torch.linalg.det(F)
            M["pF_det_le0_max"] = int((det <= 0).sum(1).max()); M["pF_det_nonfinite"] = int((~torch.isfinite(det)).sum())
        d = torch.load(data, map_location="cpu", weights_only=False)
        cfg = d["cfg"]; h = float(cfg["frame_dt"])
        sys.path.insert(0, "lib")
        from anchorflow import deform_model as DM
        m = DM.traj_mass(d, "cpu").double()
        n = min(X.shape[1], m.shape[0])
        X = X[:, :n]; m = m[:n]
        ok = torch.isfinite(X).all(2).all(1)
        T = int(ok.sum()) if bool(ok.all()) else int(torch.nonzero(~ok)[0])
        X = X[:T]
        g = torch.tensor(cfg.get("g", [0.0, 0.0, -9.8]), dtype=torch.float64)
        vol = m / float(cfg["density"])
        E_, nu = float(cfg["E"]), float(cfg["nu"]); mu = E_ / (2 * (1 + nu)); lam = E_ * nu / ((1 + nu) * (1 - 2 * nu))
        KE = torch.zeros(T, dtype=torch.float64); PE = torch.zeros(T, dtype=torch.float64); EL = torch.zeros(T, dtype=torch.float64)
        for t in range(T):
            if 0 < t:
                v = (X[t] - X[t - 1]) / h; KE[t] = 0.5 * (m * (v * v).sum(1)).sum()
            PE[t] = -(m * (X[t] * g).sum(1)).sum()
            if F is not None and t < F.shape[0]:
                Ft = F[t, :n]
                J = torch.linalg.det(Ft).clamp_min(1e-9)
                Cc = Ft.transpose(1, 2) @ Ft
                psi = 0.5 * mu * (torch.diagonal(Cc, dim1=1, dim2=2).sum(1) - 3) - mu * torch.log(J) + 0.5 * lam * torch.log(J) ** 2
                EL[t] = (vol * psi).sum()
        Et = KE + PE + EL
        sc = float(KE.max() + (PE - PE[0]).abs().max() + EL.max() + 1e-30)
        rise = float((Et[1:] - Et[0]).max()) if T > 1 else 0.0
        M["energy"] = dict(T=T, E0=float(Et[0]), rise_J=rise, rise_rel=max(rise, 0.0) / sc, KEmax=float(KE.max()), ELmax=float(EL.max()), Efinal=float(Et[-1]), scale=sc)
except Exception as e:
    import traceback
    M["energy_err"] = repr(e) + traceback.format_exc()[-400:]
if int(rep) == 1 and os.path.exists(f"{D}/q27solver.pt") and os.path.exists(f"{D}/pred.pt") and var != "q64":
    q = {"q1": 1, "q27": 27}.get(var, 8)
    kb = 800 if scene in ("bread", "plane") else 2400
    r = subprocess.run([sys.executable, "-u", "exe/q27_eig.py", "--pred", f"{D}/pred.pt", "--solver", f"{D}/q27solver.pt", "--frames", "0", "--q", str(q), "--kb", str(kb), "--out", f"{D}/eig.json"],
                       capture_output=True, text=True, timeout=900)
    try:
        E = json.load(open(f"{D}/eig.json"))
        c = E["cases"][0]; M["eig"] = dict(n_zero=c["n_zero"], n_near=c["n_near"], kb=c["kb"], all_kb_small=c["all_kb_small"], q=q)
    except Exception as e:
        M["eig_err"] = repr(e) + r.stderr[-300:]
elif int(rep) == 1 and var == "q64":
    M["eig"] = "미지원 (q27_eig.py 는 1·8·27 점만)"
json.dump(M, open(f"{D}/metrics.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: M.get(k) for k in ("frames_done", "diverged", "err", "ms", "nJneg_max")}), M.get("energy", {}).get("rise_rel"), M.get("eig"))
