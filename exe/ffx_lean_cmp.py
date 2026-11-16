# 절 CM192: lean 판 (결과 불변 진단 뺌) 이 궤적을 바꾸지 않는지 -- 같은 명령 두 번 (전체 r11·r12) 사이 차이와 전체 대 lean 차이를 프레임별로 비교
import json, subprocess, sys, torch
R = "r2:storage/result/anchorflow/lbours/final_fix/runs"
out = {}
for s in sys.argv[1:]:
    P = {}
    for tag in ("def.dump.wel_r11", "def.dump.wel_r12", "def.dump.wel.lean_r11", "def.dump.wel.lean_r12"):
        f = f"/workspace/ab/{s}_{tag}.pt"
        subprocess.run(["rclone", "copyto", f"{R}/{s}_{tag}/pred.pt", f, "-q", "--multi-thread-streams", "8"], check=True)
        d = torch.load(f, map_location="cpu", weights_only=False)
        P[tag] = (d["pred"] if isinstance(d, dict) else d).float()
    x0 = P["def.dump.wel_r11"]
    ext = float((x0[0].amax(0) - x0[0].amin(0)).norm())
    def dif(a, b):
        e = (P[a] - P[b]).norm(dim=-1).mean(-1) / ext                                  # 프레임별 평균 |dx| / 대각선
        return dict(mean=float(e.mean()), max=float(e.max()), f5=float(e[min(4, len(e) - 1)]))
    out[s] = dict(full_full=dif("def.dump.wel_r11", "def.dump.wel_r12"), lean_lean=dif("def.dump.wel.lean_r11", "def.dump.wel.lean_r12"),
                  full_lean=dif("def.dump.wel_r11", "def.dump.wel.lean_r11"), full_lean2=dif("def.dump.wel_r12", "def.dump.wel.lean_r12"))
    print(s, json.dumps(out[s]), flush=True)
    subprocess.run("rm -f /workspace/ab/*.pt", shell=True)
json.dump(out, open("/workspace/ab/ab_cmp.json", "w"), indent=1)
