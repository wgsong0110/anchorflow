"""taichi_elements(GASP 의 솔버)에 **파괴 재질**을 한 갈래 더한다.

그쪽에는 water/elastic/snow/sand 넷뿐이라 파괴가 없다. PG·i-PG 쪽 파괴 칸에
이미 쓰고 있는 **CD-MPM(GF watermelon)** 을 그대로 taichi 로 옮겨
`material_fracture = 5` 로 붙인다 -- 비연계 Cam-Clay 되돌림 + 경화 +
Borden neo-Hookean 응력. 기존 네 재질의 코드 경로는 **건드리지 않는다**.

물성 상수는 PG 파괴 칸과 같은 GF 값을 쓴다 (E 2e3, nu 0.38, 마찰각 45,
beta 1, xi 3). 솔버 전역의 E=2e6 으로는 항복면에 닿지 않아 깨지지 않는다.

  python exe/patch_ti_fracture.py            # 적용 (여러 번 돌려도 안전)
"""
from __future__ import annotations

import os
import shutil

SRC = "/home/dkta/work/taichi_elements/engine/mpm_solver.py"
MARK = "material_fracture = 5"

FUNC = '''
    @ti.func
    def camclay_borden(self, p, U, sig, V):
        """CD-MPM(GF watermelon): Cam-Clay 되돌림 + Borden 응력.

        self.Jp[p] 를 **logJp** 로 쓴다 (sand 와 같은 자리, 다른 뜻).
        되돌린 F 를 self.F[p] 에 쓰고 Kirchhoff 응력을 돌려준다.
        """
        mu = self.frac_mu
        kappa = self.frac_kappa
        M = self.frac_M
        beta = self.frac_beta
        xi = self.frac_xi
        # 경화 상태는 반드시 막아야 한다. sinh 는 f32 에서 x>88 이면 넘치고,
        # 한 번 넘치면 p0=inf -> 전부 NaN -> 입자가 격자 밖으로 날아간다.
        _hx = ti.min(xi * ti.max(-self.Jp[p], 0.0), 15.0)
        # taichi 에는 sinh 가 없다 -> (e^x - e^-x)/2
        p0 = kappa * (1e-5 + 0.5 * (ti.exp(_hx) - ti.exp(-_hx)))
        J = 1.0
        Bm = 0.0
        for i in ti.static(range(3)):
            J *= sig[i, i]
            Bm += sig[i, i] * sig[i, i]
        Bm /= 3.0
        Jn23 = ti.max(J, 1e-12) ** (-2.0 / 3.0)
        s_hat = ti.Vector([0.0, 0.0, 0.0])
        s_sq = 0.0
        for i in ti.static(range(3)):
            s_hat[i] = mu * Jn23 * (sig[i, i] * sig[i, i] - Bm)
            s_sq += s_hat[i] * s_hat[i]
        p_tr = -(kappa / 2.0 * (J - 1.0 / ti.max(J, 1e-12))) * J
        ys_c = 1.5 * (1.0 + 2.0 * beta)
        yp_h = M * M * (p_tr + beta * p0) * (p_tr - p0)
        y = ys_c * s_sq + yp_h
        p_min = beta * p0
        sig_new = sig
        if p_tr > p0:                       # 압축 꼭짓점
            Je = ti.sqrt(ti.max(-2.0 * p0 / kappa + 1.0, 1e-12))
            s = ti.min(ti.max(Je ** (1.0 / 3.0), 0.05), 20.0)
            sig_new = ti.Matrix.identity(ti.f32, 3) * s
            self.Jp[p] = ti.min(ti.max(self.Jp[p]
                                       + ti.log(ti.max(J / Je, 1e-12)),
                                       -5.0), 5.0)
        elif p_tr < -p_min:                 # 인장 꼭짓점 -- 여기서 갈라진다
            Je = ti.sqrt(ti.max(2.0 * p_min / kappa + 1.0, 1e-12))
            s = ti.min(ti.max(Je ** (1.0 / 3.0), 0.05), 20.0)
            sig_new = ti.Matrix.identity(ti.f32, 3) * s
            self.Jp[p] = ti.min(ti.max(self.Jp[p]
                                       + ti.log(ti.max(J / Je, 1e-12)),
                                       -5.0), 5.0)
        elif y >= 1e-4:                     # 항복면 위로
            s_norm = ti.max(ti.sqrt(ti.max(s_sq, 1e-20)), 1e-10)
            scale = (ti.max(J, 1e-12) ** (2.0 / 3.0) / mu
                     * ti.sqrt(ti.max(-yp_h / ys_c, 0.0)) / s_norm)
            for i in ti.static(range(3)):
                b = scale * s_hat[i] + Bm
                # 특이값을 묶는다. 작아지면 1/J 가 폭주해 응력이 터진다.
                sig_new[i, i] = ti.min(ti.max(ti.sqrt(ti.max(b, 1e-12)),
                                              0.05), 20.0)
            # 항복면 경화. 수박을 깨뜨리는 것이 이 항이다.
            p_c = (p0 - p_min) * 0.5
            q_tr = ti.sqrt(1.5) * s_norm
            d_p = p_c - p_tr
            d_q = -q_tr
            d_n = ti.sqrt(ti.max(d_p * d_p + d_q * d_q, 1e-20))
            dpn = d_p / d_n
            dqn = d_q / d_n
            Cq = M * M * (p_c + beta * p0) * (p_c - p0)
            Bq = M * M * dpn * (2.0 * p_c - p0 + beta * p0)
            Aq = M * M * dpn * dpn + (1.0 + 2.0 * beta) * dqn * dqn
            disc = ti.max(Bq * Bq - 4.0 * Aq * Cq, 0.0)
            l1 = (-Bq + ti.sqrt(disc)) / ti.max(2.0 * Aq, 1e-20)
            l2 = (-Bq - ti.sqrt(disc)) / ti.max(2.0 * Aq, 1e-20)
            p1 = p_c + l1 * dpn
            p2 = p_c + l2 * dpn
            p_fake = p2
            if (p_tr - p_c) * (p1 - p_c) > 0.0:
                p_fake = p1
            Je_f = ti.sqrt(ti.max(ti.abs(-2.0 * p_fake / kappa + 1.0), 1e-12))
            if (Je_f > 1e-4 and p0 > 1e-4 and p_tr < p0 - 1e-4
                    and p_tr > 1e-4 - p_min):
                self.Jp[p] = ti.min(ti.max(
                    self.Jp[p] + ti.log(ti.max(J / Je_f, 1e-12)),
                    -5.0), 5.0)
        self.F[p] = U @ sig_new @ V.transpose()
        Jn = 1.0
        Bm2 = 0.0
        for i in ti.static(range(3)):
            Jn *= sig_new[i, i]
            Bm2 += sig_new[i, i] * sig_new[i, i]
        Bm2 /= 3.0
        Jn = ti.min(ti.max(Jn, 1e-3), 1e3)
        Jn23b = Jn ** (-2.0 / 3.0)
        tau = ti.Matrix.zero(ti.f32, 3, 3)
        for i in ti.static(range(3)):
            _t = (mu * Jn23b * (sig_new[i, i] * sig_new[i, i] - Bm2)
                  + kappa / 2.0 * (Jn - 1.0 / Jn) * Jn)
            tau[i, i] = ti.min(ti.max(_t, -1e6), 1e6)   # 마지막 안전막
        return U @ tau @ U.transpose()

'''


def main():
    if os.environ.get("AF_REPATCH") and os.path.exists(SRC + ".orig"):
        shutil.copy(SRC + ".orig", SRC)        # 원본으로 되돌리고 다시 붙인다
    src = open(SRC).read()
    if MARK in src:
        print("[건너뜀] 이미 적용돼 있다 (AF_REPATCH=1 로 다시 붙인다)")
        return
    if not os.path.exists(SRC + ".orig"):
        shutil.copy(SRC, SRC + ".orig")

    # 1) 재질 번호
    src = src.replace("    material_stationary = 4\n",
                      "    material_stationary = 4\n"
                      "    material_fracture = 5\n", 1)
    src = src.replace("        'STATIONARY': material_stationary,\n",
                      "        'STATIONARY': material_stationary,\n"
                      "        'FRACTURE': material_fracture,\n", 1)

    # 2) 파괴 재질 상수 (GF watermelon 값). 솔버 전역 E 와 따로 둔다.
    anchor = ("        self.mu_0, self.lambda_0 = self.E / (\n"
              "            2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) *\n"
              "                                                    (1 - 2 * self.nu))\n")
    assert anchor in src, "mu_0 정의를 못 찾았다"
    src = src.replace(anchor, anchor + '''        # --- 파괴(CD-MPM) 전용 상수: PG 파괴 칸과 같은 GF watermelon 값 ---
        import math as _math
        _E, _nu = 2e3, 0.38
        self.frac_mu = _E / (2 * (1 + _nu))
        _la = _E * _nu / ((1 + _nu) * (1 - 2 * _nu))
        self.frac_kappa = 2.0 * self.frac_mu / 3.0 + _la
        _sp = _math.sin(45.0 / 180.0 * _math.pi)
        _alpha = _math.sqrt(2.0 / 3.0) * 2.0 * _sp / (3.0 - _sp)
        self.frac_M = _alpha * 3.0 / _math.sqrt(2.0 / 3.0)
        self.frac_beta = 1.0
        self.frac_xi = 3.0
''', 1)

    # 3) 되돌림 + 응력 함수
    src = src.replace("    @ti.func\n    def sand_projection(",
                      FUNC + "    @ti.func\n    def sand_projection(", 1)

    # 4) 씨뿌리기: logJp 는 0 에서 시작한다
    src = src.replace("            if material == self.material_sand:\n"
                      "                self.Jp[i] = 0\n",
                      "            if material == self.material_sand or "
                      "material == self.material_fracture:\n"
                      "                self.Jp[i] = 0\n", 1)

    # 5) p2g 의 응력을 파괴 재질에서만 덮어쓴다 (다른 재질 경로는 그대로)
    old = ("                stress = 2 * mu * (\n"
           "                    self.F[p] - U @ V.transpose()) @ self.F[p].transpose(\n"
           "                    ) + ti.Matrix.identity(ti.f32, self.dim) * la * J * (J - 1)\n")
    assert old in src, "p2g 응력 줄을 못 찾았다"
    src = src.replace(old, old +
                      "                if self.material[p] == self.material_fracture:\n"
                      "                    stress = self.camclay_borden(p, U, sig, V)\n", 1)

    open(SRC, "w").write(src)
    print(f"[적용] {SRC}  (원본은 {os.path.basename(SRC)}.orig)")


main()
