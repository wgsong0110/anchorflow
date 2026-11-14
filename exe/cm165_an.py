"""절 CM165 분석 (CPU, numpy): 판 디렉토리 <run>/esc/fNNN.npz (escwhy) 에서 윗덩어리 (가우스 점 시작 위치 z > ZC) 의 상태별 가우스 점
 ln J0 (걸음 시작 F), ln JX (시험 X = (I+∇U) F0 의 det, 반환 전), 한 걸음 변화 ln JX − ln J0, 압력 p = −K ln J (반환은 소성 상태에서 tr 를 보존하므로 반환 뒤도 같음), 
 평형 무게 압력 ρ g 깊이 (같은 (x,y) 0.04 칸 최고 점과의 z 차) 와 정답 같은 인덱스·같은 프레임 입자의 ln J (wolf.pt F) 를 비교.
사용: python3 exe/cm165_an.py <run> <wolf.pt> [--zc 0.9] [--frames 20,26,30,34]"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cm130_ptload as L
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("wolf"); ap.add_argument("--zc", type=float, default=0.9); ap.add_argument("--frames", default="20,26,30,34")
a = ap.parse_args(); np.set_printoptions(precision=4, suppress=True, linewidth=200)
E_, NU, RHO, G = 5e7, 0.3, 2000.0, 9.8; K = E_ / (3 * (1 - 2 * NU))
d = L.load(a.wolf); GF = d["F"]
def lnj_p(F):
    s = np.linalg.svd(F.reshape(-1, 3, 3).astype(np.float64), compute_uv=False); return np.log(np.maximum(s.prod(1), 1e-9))
gl = {}
Q = (10, 50, 90)
def q(x): return np.percentile(x, Q) if len(x) else np.zeros(3)
for fr in [int(v) for v in a.frames.split(",")]:
    z = np.load(os.path.join(a.run, "esc", "f%03d.npz" % fr))
    xq = z["xq0"]; S = z["QS"][0]; st = S[:, 3].astype(int); JX = S[:, 4]; F0 = z["FQ0"]; vq = z["Vq"]; mq = z["mq"]
    J0 = np.linalg.det(F0.reshape(-1, 3, 3).astype(np.float64))
    ok = (mq > 0) & (J0 > 0) & (JX > 0)
    key = np.floor(xq[:, 0] / 0.04).astype(np.int64) * 1000 + np.floor(xq[:, 1] / 0.04).astype(np.int64)
    top = {}
    for k_, z_ in zip(key[ok], xq[ok, 2]): top[k_] = max(top.get(k_, -9), z_)
    depth = np.array([top.get(k_, z_) - z_ for k_, z_ in zip(key, xq[:, 2])])
    sel = ok & (xq[:, 2] > a.zc)
    l0 = np.log(np.where(ok, J0, 1)); lx = np.log(np.where(ok, JX, 1))
    p0 = -K * l0 / 1e3; px = -K * lx / 1e3; pw = RHO * G * depth / 1e3
    print(f"== f{fr}: z > {a.zc} 가우스 점 {sel.sum()} (시험 상태 몫 탄성/꼬리/꼭짓점 {np.mean(st[sel]==0):.2f}/{np.mean(st[sel]==1):.2f}/{np.mean(st[sel]==2):.2f})")
    for s_, nm in ((0, "탄성"), (1, "꼬리 (소성)"), (2, "꼭짓점")):
        m_ = sel & (st == s_)
        if m_.sum() < 5: continue
        print(f"  상태 {nm}: 점 {m_.sum()} | ln J0 {q(l0[m_])} ln JX {q(lx[m_])} 한 걸음 ΔlnJ {q((lx-l0)[m_])} | 시작 압력 kPa {q(p0[m_])} 시험 압력 {q(px[m_])} | 무게 압력 {q(pw[m_])} | 시험 압력/무게 압력 중앙값 {np.median(px[m_]/np.maximum(pw[m_],0.5)):.1f}")
    # 정답: 같은 프레임 입자 ln J (모든 입자 부표본), 압력
    gj = lnj_p(GF[fr][::20]); print(f"  정답 (f{fr}, 입자 1/20) ln J {q(gj)} 압력 kPa {q(-K*gj/1e3)}, |ln J|>0.003 몫 {np.mean(np.abs(gj)>3e-3):.3f} | 우리 소성 점 (z>{a.zc}) |ln JX|>0.003 몫 {np.mean(np.abs(lx[sel&(st==1)])>3e-3) if (sel&(st==1)).any() else 0:.3f}")
