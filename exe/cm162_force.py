"""절 CM162.2 항별 분해 (CPU, numpy): 판 디렉토리 <run>/esc/fNNN.npz (escwhy) + <run>/tear/fNNN.npz (fem_tear2: 노드 탄성 기울기 fel = ∂Ψ/∂y (복원력 = −fel), 가우스 점 코시 응력)
에서 윗덩어리 (걸음 시작 z > ZC 인 노드) 의 수직 운동 방정식을 걸음마다 항별로: 관측 가속 a_obs = Σ m Δv_z/Δt / M, 중력 −9.8, 탄성 복원력 −Σ fel_z / M, 나머지 (장벽·관성 갱신·미수렴 잔차) = a_obs − g − 탄성.
또 높이 z0 위 전체 노드의 순 탄성력/무게 (받침 비율) 층 프로파일 -> 낙하를 붙잡는 높이, 그 층 가우스 점의 σ_zz·상태 (0 탄성 1 꼬리 2 꼭짓점)·J, 구속 kb<1 노드 몫, 마지막 반복 잔차 |g0|/무게.
사용: python3 exe/cm162_force.py <run> [--zc 0.9] [--f0 20 --f1 34] [--pn 0.48]"""
import argparse, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--zc", type=float, default=0.9); ap.add_argument("--f0", type=int, default=20); ap.add_argument("--f1", type=int, default=34); ap.add_argument("--pn", type=float, default=0.48)
a = ap.parse_args(); np.set_printoptions(precision=3, suppress=True, linewidth=200)
G = 9.8
print(f"윗덩어리 = 걸음 시작 z > {a.zc} 노드. 열: 프레임 | 노드 수·질량 | a_obs | 탄성 a_el = −Σfel_z/M | 중력 | 나머지 a_obs−g−a_el | kb<1 몫 | 잔차 |g0| 합/무게 합")
rows = []
for fr in range(a.f0, a.f1 + 1):
    ef = os.path.join(a.run, "esc", "f%03d.npz" % fr); tf = os.path.join(a.run, "tear", "f%03d.npz" % fr)
    if not (os.path.exists(ef) and os.path.exists(tf)): continue
    z = np.load(ef); t = np.load(tf)
    nid = z["nid"].astype(np.int64); IP = z["IP"]; A = z["A"]; nb1, nb2 = IP[7], IP[8]
    pos = (np.stack([nid // (nb1 * nb2), (nid // nb2) % nb1, nid % nb2], 1) + IP[0:3]) @ A.T + z["U0"][0]
    m = z["mn"].astype(np.float64); dt = float(z["h"]) / int(z["K"])
    aid = t["aid"].astype(np.int64)
    if len(aid) != len(nid) or not np.array_equal(aid, nid): print(f"f{fr}: 노드 번호 불일치 (esc {len(nid)} tear {len(aid)})"); continue
    fel = t["fel"].astype(np.float64)[:, 2]; ael_n = -fel
    big = m > 1e-2 * np.median(m)
    c = big & (pos[:, 2] > a.zc)
    if c.sum() < 5: continue
    M = m[c].sum(); aobs = (m[c] * (z["V1"][0][c, 2] - z["V0"][0][c, 2]) / dt).sum() / M
    ael = ael_n[c].sum() / M
    kb = (z["KB"][0][c] < 1).mean(); r0 = np.linalg.norm(z["G0"][0][c], axis=1).sum() / (m[c].sum() * G)
    Ut_=z["Ut"][0]; Y_=z["Y"][0]; inert=((m[c]*((Y_-Ut_)[c,2]/dt**2)).sum()/M)
    rows.append((fr, aobs, ael))
    print(f"f{fr} | {c.sum()} 노드 {M:.2f} kg | {aobs:+.2f} | {ael:+.2f} | {-G:+.2f} | {aobs + G - ael:+.2f} | {kb:.2f} | {r0:.2f} | (Y−Ũ)_z/dt² {inert:+.2f}")
    if fr in (22, 26, 30, 34):
        print("   층 프로파일 (z0 위 전체 노드): z0 | 위 질량 | 순 탄성력/무게 | 위 질량 가중 평균 v_z | 그 층 (z0±0.03) 가우스 점 σ_zz 중앙값·J 중앙값·상태 몫 (탄성/꼬리/꼭짓점)")
        q = t["q"]; qz = t["q_pos"][:, 2]
        for z0 in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
            up = big & (pos[:, 2] > z0)
            if up.sum() < 5: continue
            Mu = m[up].sum(); ratio = ael_n[up].sum() / (Mu * G)
            vz = (m[up] * z["V0"][0][up, 2]).sum() / Mu
            dz = 0.04; s_ = np.abs(qz - z0) < dz
            if s_.sum() > 5:
                st = q[s_, 10]; sz = q[s_, 2]; vq = q[s_, 11]
                Fz = [-(sz[st == k] * vq[st == k]).sum() / (2 * dz) for k in (0, 1, 2)]
                txt = f"응력으로 본 받침력/무게 (탄성·꼬리·꼭짓점) {Fz[0]/(Mu*G):+.2f} {Fz[1]/(Mu*G):+.2f} {Fz[2]/(Mu*G):+.2f} 합 {sum(Fz)/(Mu*G):+.2f} · 상태 몫 {(st==0).mean():.2f}/{(st==1).mean():.2f}/{(st==2).mean():.2f} · J 중앙값 {np.median(q[s_,9]):.3f} (점 {s_.sum()})"
            else: txt = "점 없음"
            print(f"      {z0:.1f} | {Mu:6.2f} kg | 노드 탄성력/무게 {ratio:+.2f} | v_z {vz:+.2f} | {txt}")
if rows:
    r = np.array(rows); print("평균 (f%d~%d): a_obs %+.2f a_el %+.2f 나머지 %+.2f  (g −9.8)" % (r[0, 0], r[-1, 0], r[:, 1].mean(), r[:, 2].mean(), (r[:, 1] + G - r[:, 2]).mean()))
