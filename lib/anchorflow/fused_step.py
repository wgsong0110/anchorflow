"""증분 포텐셜 평가(에너지+기울기)를 **커널 한 번**으로 (근사 없음).

입자 하나를 프로그램 하나가 맡는다. 그 입자가 속한 사면체의 네 꼭짓점 변위를
모아

    J = I + sum_i u_i (x) b_i,   F = J F^n
    psi = mu sum (sigma_i - 1)^2 + lam/2 (det F - 1)^2        (고정 코로테이션)
    dpsi/dF = 2 mu (F - R) + lam (J_F - 1) J_F F^-T           (R = 극분해 회전)

를 닫힌 식으로 계산하고, 사슬규칙으로 노드 기울기

    dE/du_i = V_p (dpsi/dF) (F^n^T b_i)

를 만들어 atomic 으로 더한다. 관성항도 같은 커널에서 처리한다.
3x3 대칭 고유값은 삼각함수 닫힌 식, C^{1/2} 은 Franca 공식을 쓴다 (고유벡터를
구하지 않는다).
"""
from __future__ import annotations

import math
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                    # pragma: no cover
    HAVE_TRITON = False

if HAVE_TRITON:

    @triton.jit
    def _step_kernel(DPN, ROWS, LAM, B, FN, MASS, VOL, TGT,
                     MU, LAM_E, EOUT, GRAD, NP, BLOCK: tl.constexpr):
        p = tl.program_id(0)
        if p < NP:
            # --- 노드 변위 모으기 ---------------------------------------
            r0 = tl.load(ROWS + p * 4 + 0)
            r1 = tl.load(ROWS + p * 4 + 1)
            r2 = tl.load(ROWS + p * 4 + 2)
            r3 = tl.load(ROWS + p * 4 + 3)
            u00 = tl.load(DPN + r0 * 3 + 0); u01 = tl.load(DPN + r0 * 3 + 1)
            u02 = tl.load(DPN + r0 * 3 + 2)
            u10 = tl.load(DPN + r1 * 3 + 0); u11 = tl.load(DPN + r1 * 3 + 1)
            u12 = tl.load(DPN + r1 * 3 + 2)
            u20 = tl.load(DPN + r2 * 3 + 0); u21 = tl.load(DPN + r2 * 3 + 1)
            u22 = tl.load(DPN + r2 * 3 + 2)
            u30 = tl.load(DPN + r3 * 3 + 0); u31 = tl.load(DPN + r3 * 3 + 1)
            u32 = tl.load(DPN + r3 * 3 + 2)
            b00 = tl.load(B + p * 12 + 0); b01 = tl.load(B + p * 12 + 1)
            b02 = tl.load(B + p * 12 + 2)
            b10 = tl.load(B + p * 12 + 3); b11 = tl.load(B + p * 12 + 4)
            b12 = tl.load(B + p * 12 + 5)
            b20 = tl.load(B + p * 12 + 6); b21 = tl.load(B + p * 12 + 7)
            b22 = tl.load(B + p * 12 + 8)
            b30 = tl.load(B + p * 12 + 9); b31 = tl.load(B + p * 12 + 10)
            b32 = tl.load(B + p * 12 + 11)
            # G = sum_i u_i b_i^T
            g00 = u00 * b00 + u10 * b10 + u20 * b20 + u30 * b30
            g01 = u00 * b01 + u10 * b11 + u20 * b21 + u30 * b31
            g02 = u00 * b02 + u10 * b12 + u20 * b22 + u30 * b32
            g10 = u01 * b00 + u11 * b10 + u21 * b20 + u31 * b30
            g11 = u01 * b01 + u11 * b11 + u21 * b21 + u31 * b31
            g12 = u01 * b02 + u11 * b12 + u21 * b22 + u31 * b32
            g20 = u02 * b00 + u12 * b10 + u22 * b20 + u32 * b30
            g21 = u02 * b01 + u12 * b11 + u22 * b21 + u32 * b31
            g22 = u02 * b02 + u12 * b12 + u22 * b22 + u32 * b32
            j00 = 1.0 + g00; j01 = g01; j02 = g02
            j10 = g10; j11 = 1.0 + g11; j12 = g12
            j20 = g20; j21 = g21; j22 = 1.0 + g22
            # --- F = J Fn ------------------------------------------------
            n00 = tl.load(FN + p * 9 + 0); n01 = tl.load(FN + p * 9 + 1)
            n02 = tl.load(FN + p * 9 + 2); n10 = tl.load(FN + p * 9 + 3)
            n11 = tl.load(FN + p * 9 + 4); n12 = tl.load(FN + p * 9 + 5)
            n20 = tl.load(FN + p * 9 + 6); n21 = tl.load(FN + p * 9 + 7)
            n22 = tl.load(FN + p * 9 + 8)
            f00 = j00 * n00 + j01 * n10 + j02 * n20
            f01 = j00 * n01 + j01 * n11 + j02 * n21
            f02 = j00 * n02 + j01 * n12 + j02 * n22
            f10 = j10 * n00 + j11 * n10 + j12 * n20
            f11 = j10 * n01 + j11 * n11 + j12 * n21
            f12 = j10 * n02 + j11 * n12 + j12 * n22
            f20 = j20 * n00 + j21 * n10 + j22 * n20
            f21 = j20 * n01 + j21 * n11 + j22 * n21
            f22 = j20 * n02 + j21 * n12 + j22 * n22
            # --- C = F^T F 와 그 고유값 (닫힌 식) -------------------------
            c00 = f00 * f00 + f10 * f10 + f20 * f20
            c01 = f00 * f01 + f10 * f11 + f20 * f21
            c02 = f00 * f02 + f10 * f12 + f20 * f22
            c11 = f01 * f01 + f11 * f11 + f21 * f21
            c12 = f01 * f02 + f11 * f12 + f21 * f22
            c22 = f02 * f02 + f12 * f12 + f22 * f22
            q = (c00 + c11 + c22) / 3.0
            p1 = (c00 - q) * (c00 - q) + (c11 - q) * (c11 - q) \
                + (c22 - q) * (c22 - q) + 2.0 * (c01 * c01 + c02 * c02
                                                 + c12 * c12)
            pp = tl.sqrt(p1 / 6.0 + 1e-30)
            d00 = (c00 - q) / pp; d01 = c01 / pp; d02 = c02 / pp
            d11 = (c11 - q) / pp; d12 = c12 / pp; d22 = (c22 - q) / pp
            det = (d00 * (d11 * d22 - d12 * d12)
                   - d01 * (d01 * d22 - d12 * d02)
                   + d02 * (d01 * d12 - d11 * d02)) * 0.5
            det = tl.minimum(tl.maximum(det, -1.0), 1.0)
            phi = tl.math.acos(det) / 3.0
            e1 = q + 2.0 * pp * tl.math.cos(phi)
            e3 = q + 2.0 * pp * tl.math.cos(phi + 2.0943951023931953)
            e2 = 3.0 * q - e1 - e3
            s1 = tl.sqrt(tl.maximum(e1, 1e-12))
            s2 = tl.sqrt(tl.maximum(e2, 1e-12))
            s3 = tl.sqrt(tl.maximum(e3, 1e-12))
            # --- C^{1/2} (Franca) 와 R = F C^{-1/2} ----------------------
            ps = s1 + s2 + s3
            qs = s1 * s2 + s2 * s3 + s3 * s1
            rs = s1 * s2 * s3
            dn = ps * qs - rs + 1e-30
            # C² (대칭)
            k00 = c00 * c00 + c01 * c01 + c02 * c02
            k01 = c00 * c01 + c01 * c11 + c02 * c12
            k02 = c00 * c02 + c01 * c12 + c02 * c22
            k11 = c01 * c01 + c11 * c11 + c12 * c12
            k12 = c01 * c02 + c11 * c12 + c12 * c22
            k22 = c02 * c02 + c12 * c12 + c22 * c22
            t2 = ps * ps - qs
            h00 = (-k00 + t2 * c00 + ps * rs) / dn
            h01 = (-k01 + t2 * c01) / dn
            h02 = (-k02 + t2 * c02) / dn
            h11 = (-k11 + t2 * c11 + ps * rs) / dn
            h12 = (-k12 + t2 * c12) / dn
            h22 = (-k22 + t2 * c22 + ps * rs) / dn
            # S^{-1} = adj(S)/det(S), det(S) = rs
            a00 = (h11 * h22 - h12 * h12) / rs
            a01 = -(h01 * h22 - h12 * h02) / rs
            a02 = (h01 * h12 - h11 * h02) / rs
            a11 = (h00 * h22 - h02 * h02) / rs
            a12 = -(h00 * h12 - h01 * h02) / rs
            a22 = (h00 * h11 - h01 * h01) / rs
            r00 = f00 * a00 + f01 * a01 + f02 * a02
            r01 = f00 * a01 + f01 * a11 + f02 * a12
            r02 = f00 * a02 + f01 * a12 + f02 * a22
            r10 = f10 * a00 + f11 * a01 + f12 * a02
            r11 = f10 * a01 + f11 * a11 + f12 * a12
            r12 = f10 * a02 + f11 * a12 + f12 * a22
            r20 = f20 * a00 + f21 * a01 + f22 * a02
            r21 = f20 * a01 + f21 * a11 + f22 * a12
            r22 = f20 * a02 + f21 * a12 + f22 * a22
            # --- psi 와 dpsi/dF ------------------------------------------
            jf = s1 * s2 * s3
            mu = MU
            lamv = LAM_E
            vol = tl.load(VOL + p)
            psi = mu * ((s1 - 1.0) * (s1 - 1.0) + (s2 - 1.0) * (s2 - 1.0)
                        + (s3 - 1.0) * (s3 - 1.0)) \
                + 0.5 * lamv * (jf - 1.0) * (jf - 1.0)
            # F^{-T} * det = cofactor(F)
            cf00 = f11 * f22 - f12 * f21
            cf01 = f12 * f20 - f10 * f22
            cf02 = f10 * f21 - f11 * f20
            cf10 = f02 * f21 - f01 * f22
            cf11 = f00 * f22 - f02 * f20
            cf12 = f01 * f20 - f00 * f21
            cf20 = f01 * f12 - f02 * f11
            cf21 = f02 * f10 - f00 * f12
            cf22 = f00 * f11 - f01 * f10
            cl = lamv * (jf - 1.0)
            p00 = 2.0 * mu * (f00 - r00) + cl * cf00
            p01 = 2.0 * mu * (f01 - r01) + cl * cf10
            p02 = 2.0 * mu * (f02 - r02) + cl * cf20
            p10 = 2.0 * mu * (f10 - r10) + cl * cf01
            p11 = 2.0 * mu * (f11 - r11) + cl * cf11
            p12 = 2.0 * mu * (f12 - r12) + cl * cf21
            p20 = 2.0 * mu * (f20 - r20) + cl * cf02
            p21 = 2.0 * mu * (f21 - r21) + cl * cf12
            p22 = 2.0 * mu * (f22 - r22) + cl * cf22
            # --- 관성 ------------------------------------------------------
            l0 = tl.load(LAM + p * 4 + 0); l1 = tl.load(LAM + p * 4 + 1)
            l2 = tl.load(LAM + p * 4 + 2); l3 = tl.load(LAM + p * 4 + 3)
            up0 = l0 * u00 + l1 * u10 + l2 * u20 + l3 * u30
            up1 = l0 * u01 + l1 * u11 + l2 * u21 + l3 * u31
            up2 = l0 * u02 + l1 * u12 + l2 * u22 + l3 * u32
            mh = tl.load(MASS + p)
            t0 = tl.load(TGT + p * 3 + 0); t1 = tl.load(TGT + p * 3 + 1)
            t2v = tl.load(TGT + p * 3 + 2)
            d0 = up0 - t0; d1 = up1 - t1; d2 = up2 - t2v
            e_in = 0.5 * mh * (d0 * d0 + d1 * d1 + d2 * d2)
            tl.store(EOUT + p, vol * psi + e_in)
            # --- 노드로 기울기 흩뿌리기 ------------------------------------
            for i in range(4):
                bi0 = tl.load(B + p * 12 + i * 3 + 0)
                bi1 = tl.load(B + p * 12 + i * 3 + 1)
                bi2 = tl.load(B + p * 12 + i * 3 + 2)
                # w = Fn^T b_i
                w0 = n00 * bi0 + n10 * bi1 + n20 * bi2
                w1 = n01 * bi0 + n11 * bi1 + n21 * bi2
                w2 = n02 * bi0 + n12 * bi1 + n22 * bi2
                li = tl.load(LAM + p * 4 + i)
                gx = vol * (p00 * w0 + p01 * w1 + p02 * w2) + mh * li * d0
                gy = vol * (p10 * w0 + p11 * w1 + p12 * w2) + mh * li * d1
                gz = vol * (p20 * w0 + p21 * w1 + p22 * w2) + mh * li * d2
                ri = tl.load(ROWS + p * 4 + i)
                tl.atomic_add(GRAD + ri * 3 + 0, gx)
                tl.atomic_add(GRAD + ri * 3 + 1, gy)
                tl.atomic_add(GRAD + ri * 3 + 2, gz)


class FusedStep(torch.autograd.Function):
    """E(dpn) = 탄성 + 관성. 순전파에서 기울기까지 만들어 둔다 (커널 1 회)."""

    @staticmethod
    def forward(ctx, dpn, rows, lam, b, Fn, mass, vol, tgt, mu, lam_e):
        n = rows.shape[0]
        e = torch.empty(n, device=dpn.device, dtype=dpn.dtype)
        gr = torch.zeros_like(dpn)
        _step_kernel[(n,)](dpn, rows, lam, b, Fn, mass, vol, tgt,
                           float(mu), float(lam_e), e, gr, n, BLOCK=1)
        ctx.save_for_backward(gr)
        return e.sum()

    @staticmethod
    def backward(ctx, go):
        (gr,) = ctx.saved_tensors
        return (go * gr, None, None, None, None, None, None, None, None, None)


def energy(dpn, rows, lam, b, Fn, mass, vol, tgt, mu, lam_e):
    return FusedStep.apply(dpn, rows.to(torch.int32).contiguous(),
                           lam.contiguous(), b.contiguous(), Fn.contiguous(),
                           mass.contiguous(), vol.contiguous(),
                           tgt.contiguous(), mu, lam_e)
