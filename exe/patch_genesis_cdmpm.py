"""Genesis 의 MPM 에 **파괴되는 재질**(CD-MPM / 비연관 Cam-Clay)을 추가한다.

RAF 의 얼개 -- 3DGS 를 입자로 추상화하고, 입자 시뮬을 돌리고, 결과로 3DGS 를
스키닝 -- 는 시뮬레이터를 갈아 끼울 수 있게 되어 있다. 막고 있는 것은 얼개가
아니라 Genesis 의 재질 목록이다: Elastic / ElastoPlastic(von Mises) / Sand /
Snow / Liquid / Muscle 에 **손상·연화 항이 없어** 항복해도 재료가 약해지지 않고,
그래서 끊어지는 대신 늘어나거나 퍼진다.

그래서 GaussianFluent 에서 확인한 CD-MPM(Wolper et al.) 을 Genesis 재질로 옮긴다.
핵심은 **항복면 위의 경화 블록**이다 -- 항복한 자리에서 logJp 가 커지면 p0 이
내려가 그 자리가 물러지고, 그 되먹임이 균열을 만든다. 그게 빠지면 beta 를 어떻게
줘도 이웃 이탈이 0.00% 로 나온다 (직접 겪었다).

Genesis 의 재질 인터페이스가 마침 이 모델에 맞는다: `Jp` 가 입자별 소성 상태이고
`_default_Jp` 로 NACC 의 alpha_0(-0.04)를 줄 수 있다.

  python exe/patch_genesis_cdmpm.py --genesis <site-packages/genesis>
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--genesis", required=True)
a = ap.parse_args()

SRC = '''import math
from typing import Any

import quadrants as qd
from pydantic import Field

import genesis as gs
from genesis.typing import PositiveFloat, ValidFloat

from .base import Base, SamplerType


@qd.data_oriented
class CDMPM(Base):
    """CD-MPM (Wolper et al.) -- 비연관 Cam-Clay + neo-Hookean(Boarden).

    GaussianFluent 의 material 7 을 그대로 옮긴 것. 항복면 위에서 logJp 를 키워
    p0 을 끌어내리는 경화가 들어 있어 **실제로 갈라진다**. 나머지 Genesis 재질에는
    이 되먹임이 없어 소성 신장까지만 간다.

    Parameters
    ----------
    E, nu, rho : 탄성과 밀도
    friction_angle : 항복면 기울기 M 을 정한다 (도). 기본 45
    beta : 인장 쪽 항복면 크기. 작을수록 잘 끊어진다. 기본 1
    xi : 경화 지수. 기본 3
    hardening : 1 이면 경화를 켠다. 기본 1
    alpha_0 : logJp 초기값. 기본 -0.04
    """

    sampler: SamplerType = "pbs"
    friction_angle: PositiveFloat = 45.0
    beta: ValidFloat = 1.0
    xi: ValidFloat = 3.0
    hardening: ValidFloat = 1.0
    alpha_0: ValidFloat = -0.04

    kappa: ValidFloat = Field(default=0.0, exclude=True)
    M_cd: ValidFloat = Field(default=0.0, exclude=True)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._default_Jp = self.alpha_0
        self.kappa = 2.0 * self.mu / 3.0 + self.lam
        sin_phi = math.sin(math.radians(self.friction_angle))
        alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
        self.M_cd = alpha * 3.0 / math.sqrt(2.0 / 3.0)
        self.update_F_S_Jp = self._update_F_S_Jp_cdmpm
        self.update_stress = self._update_stress_cdmpm

    @qd.func
    def _update_F_S_Jp_cdmpm(self, J, F_tmp, U, S, V, Jp):
        s0 = max(S[0, 0], gs.qd_float(0.0))
        s1 = max(S[1, 1], gs.qd_float(0.0))
        s2 = max(S[2, 2], gs.qd_float(0.0))
        logJp = Jp

        # p0 = kappa * (1e-5 + sinh(xi * max(-logJp, 0)))
        z = self.xi * max(-logJp, gs.qd_float(0.0))
        p0 = self.kappa * (1e-5 + 0.5 * (qd.exp(z) - qd.exp(-z)))

        Jd = s0 * s1 * s2
        b0, b1, b2 = s0 * s0, s1 * s1, s2 * s2
        bm = (b0 + b1 + b2) / 3.0
        jp = qd.pow(Jd, -2.0 / 3.0)
        sh0 = self.mu * jp * (b0 - bm)
        sh1 = self.mu * jp * (b1 - bm)
        sh2 = self.mu * jp * (b2 - bm)
        p_tr = -(self.kappa / 2.0 * (Jd - 1.0 / Jd)) * Jd

        ysc = (6.0 - 3.0) / 2.0 * (1.0 + 2.0 * self.beta)
        yph = self.M_cd * self.M_cd * (p_tr + self.beta * p0) * (p_tr - p0)
        ssq = sh0 * sh0 + sh1 * sh1 + sh2 * sh2
        y = ysc * ssq + yph
        p_min = self.beta * p0

        f0, f1, f2 = s0, s1, s2
        lj = logJp
        if p_tr > p0:
            Je = qd.sqrt(-2.0 * p0 / self.kappa + 1.0)
            f0 = qd.pow(Je, 1.0 / 3.0)
            f1 = f0
            f2 = f0
            if self.hardening > 0.5:
                lj = logJp + qd.log(Jd / Je)
        elif p_tr < -p_min:
            Je = qd.sqrt(2.0 * p_min / self.kappa + 1.0)
            f0 = qd.pow(Je, 1.0 / 3.0)
            f1 = f0
            f2 = f0
            if self.hardening > 0.5:
                lj = logJp + qd.log(Jd / Je)
        elif y >= 1e-4:
            sn = max(qd.sqrt(ssq), gs.qd_float(1e-10))
            sf = qd.sqrt(-yph / ysc)
            sc = qd.pow(Jd, 2.0 / 3.0) / self.mu * sf / sn
            f0 = qd.sqrt(max(sc * sh0 + bm, gs.qd_float(1e-12)))
            f1 = qd.sqrt(max(sc * sh1 + bm, gs.qd_float(1e-12)))
            f2 = qd.sqrt(max(sc * sh2 + bm, gs.qd_float(1e-12)))
            # 항복면 위의 경화 -- 이게 균열을 만든다
            if (self.hardening > 0.5 and p0 > 1e-4 and p_tr < p0 - 1e-4
                    and p_tr > 1e-4 - p_min):
                pc = (p0 - p_min) * 0.5
                qt = qd.sqrt((6.0 - 3.0) / 2.0) * sn
                dp = pc - p_tr
                dq = -qt
                dn = max(qd.sqrt(dp * dp + dq * dq), gs.qd_float(1e-10))
                dp = dp / dn
                Cq = self.M_cd * self.M_cd * (pc + self.beta * p0) * (pc - p0)
                Bq = self.M_cd * self.M_cd * dp * (2.0 * pc - p0 + self.beta * p0)
                Aq = (self.M_cd * self.M_cd * dp * dp
                      + (1.0 + 2.0 * self.beta) * dq * dq)
                disc = Bq * Bq - 4.0 * Aq * Cq
                l1 = (-Bq + qd.sqrt(disc)) / (2.0 * Aq)
                l2 = (-Bq - qd.sqrt(disc)) / (2.0 * Aq)
                p1 = pc + l1 * dp
                p2 = pc + l2 * dp
                pf = p2
                if (p_tr - pc) * (p1 - pc) > 0.0:
                    pf = p1
                jef = qd.sqrt(abs(-2.0 * pf / self.kappa + 1.0))
                if jef > 1e-4:
                    lj = logJp + qd.log(Jd / jef)

        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        S_new[0, 0] = f0
        S_new[1, 1] = f1
        S_new[2, 2] = f2
        F_new = U @ S_new @ V.transpose()
        return F_new, S_new, lj

    @qd.func
    def _update_stress_cdmpm(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        # kirchoff_stress_neoHookeanBoarden -- U/S/V 를 쓰지 않는다
        Jn = F_new.determinant()
        B = F_new @ F_new.transpose()
        btr = B[0, 0] + B[1, 1] + B[2, 2]
        devB = B - qd.Matrix.identity(gs.qd_float, 3) * (btr / 3.0)
        prime = self.kappa / 2.0 * (Jn - 1.0 / Jn)
        stress = (self.mu * qd.pow(Jn, -2.0 / 3.0) * devB
                  + qd.Matrix.identity(gs.qd_float, 3) * (Jn * prime))
        return stress
'''

d = os.path.join(a.genesis, "engine", "materials", "MPM")
p = os.path.join(d, "cdmpm.py")
open(p, "w").write(SRC)
print(f"넣었다: {p}")

ip = os.path.join(d, "__init__.py")
s = open(ip).read()
if "from .cdmpm import CDMPM" not in s:
    s = s.replace("from .base import Base", "from .base import Base\nfrom .cdmpm import CDMPM", 1)
    open(ip, "w").write(s)
    print(f"등록: {ip}")
print("GENESIS_CDMPM_OK")
