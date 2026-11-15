"""[paper_remeasure2] R2 lbours/paper_remeasure2/stab_<A..F>/grid.json 를 모아 하나로: python exe/pr2_collect.py <출력 json>
레고는 뺌. 움직이지 않은 (뉴턴·PCG 반복 0, 오차가 정답 변위 곡선과 같음) 조합은 그림 대상에서 빼고 사유를 invalid.json 에 남김."""
import json
import subprocess
import sys

R = "r2:storage/result/anchorflow/lbours/paper_remeasure2"
G = []
for T in ["A", "B", "C", "D", "E", "F", "H", "D2", "F2", "H2", "F3", "E2"]:                           # 뒤 샤드가 앞 샤드를 이김 (D2·F2·H2 = 회전 구동기 구현 뒤 다시 잰 것)
    r = subprocess.run(["rclone", "cat", f"{R}/stab_{T}/grid.json"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        for g in json.loads(r.stdout):
            g["shard"] = T
            G.append(g)
best = {}
for g in G:                                                                             # 같은 (장면, 방법, N) 이 여러 샤드에 있으면 완주가 많은 쪽 (C 의 phystwin ficus 는 절대 경로 문제로 실패 -> H 에서 다시 잼)
    k = (g["scene"], g["method"], g["N"])
    if k not in best or g["ok"] > best[k]["ok"] or (g["ok"] > 0 and g["shard"] in ("D2", "F2", "H2", "F3", "E2")):
        best[k] = g
G = list(best.values())
G = [g for g in G if g["scene"] != "L000"]
json.dump(G, open(sys.argv[1].replace(".json", "_backup_with_simplicits.json"), "w"), ensure_ascii=False, indent=1)   # 원본 백업 (R2 에는 그대로 있음)
G = [g for g in G if g["method"] not in ("simplicits",)]                          # 사용자 지시 11-15: Simplicits 는 모든 실험에서 뺌
INVALID = {}                                                                  # 자동 판정만 (아래 is_static)
# 입자가 움직이지 않은 조합 자동 판정: 오차가 정답 변위 곡선만으로 정해지는 값 (구동기 미적용 = 정지 예측) 과 같음 (vasedeck 2.13193 %, plane 0.89949 %)
STATIC = {"vasedeck": 2.1319291, "plane": 0.8994940}
def is_static(g):
    s_ = STATIC.get(g["scene"])
    return s_ is not None and g["err_all"] and all(abs(e - s_) < 1e-4 * s_ + 1e-6 for e in g["err_all"])
for g in G:
    if is_static(g) and (g["scene"], g["method"]) not in INVALID:
        INVALID[(g["scene"], g["method"])] = "구동기 (enforce_particle_velocity_rotation) 가 적용되지 않아 입자가 움직이지 않음 (오차가 정답 변위만으로 정해진 값과 같음)"
ok = [g for g in G if (g["scene"], g["method"]) not in INVALID]
bad = [dict(scene=s, method=m, why=w) for (s, m), w in INVALID.items()]
json.dump(ok, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
json.dump(dict(invalid=bad, all=G), open(sys.argv[1].replace(".json", "_all.json"), "w"), ensure_ascii=False, indent=1)
print(len(ok), "점", len(G) - len(ok), "점 제외")
