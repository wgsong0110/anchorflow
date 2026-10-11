"""롤아웃 빠른 경로(fast_roll) 의 **입자 단위 융합 커널** (Triton).

느린 경로에서 GPU 시간의 대부분은 입자 31 만 개에 대해 [N,4,30] 짜리 중간
텐서를 만들고(cat·곱·빼기) index_add 로 뿌리는 데 들었다. 그 중간 텐서는 한 번
쓰고 버리는 것이라, 입자 하나를 스레드 하나가 맡아 레지스터에서 계산하고 곧장
atomic 으로 더하면 메모리 왕복이 사라진다.

  locate_mark  격자 좌표 y -> 사면체 4 꼭짓점 평탄색인·barycentric·축순열,
               활성 노드 표시 (SX.locate 와 같은 연산)
  p2g_first    압축 색인 rows 확정 + 1 차 누적 g1 (가중치·x·X·v·개수, 11 칸)
               + 노드별 14 비트 간선표 칠하기
  p2g_second   중심 둘레 2 차 누적 (대칭 둘은 6 칸씩만 -> 24 칸)
  g2p          노드 변위(와 tau 탄젠트) -> 입자 위치·속도·야코비안 I + grad u

수치: 곱·차는 원래 torch 연산과 같은 순서로 쓴다. 다른 것은 (1) atomic 덧셈의
순서 (원래 index_add 도 atomic 이라 실행마다 다르다), (2) 대칭 2 차 모멘트를
위 삼각만 더해 거울로 채우는 것 (원래도 (i,j) 와 (j,i) 의 입자별 값은 같다),
(3) 컴파일러의 곱-덧셈 FMA 융합 -- 모두 1 ulp 수준이다.
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                    # pragma: no cover
    HAVE_TRITON = False

BLOCK = 256

if HAVE_TRITON:

    @triton.jit
    def _cls_of(b):
        """사슬 걸음 축 비트합 b -> 양의 간선 종류 (fast_roll._BIT2CLS)."""
        return tl.where(b == 1, 0, tl.where(b == 2, 1, tl.where(b == 4, 2, tl.where(
            b == 3, 3, tl.where(b == 5, 4, tl.where(b == 6, 5, 6))))))

    @triton.jit
    def _locate_mark(Y, NN, FLAT, LAM, PERM, MARK, N, CC, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        p = pid * BLOCK + tl.arange(0, BLOCK)
        m = p < N
        n0 = tl.load(NN + 0)
        n1 = tl.load(NN + 1)
        n2 = tl.load(NN + 2)
        y0 = tl.load(Y + p * 3 + 0, mask=m, other=0.0)
        y1 = tl.load(Y + p * 3 + 1, mask=m, other=0.0)
        y2 = tl.load(Y + p * 3 + 2, mask=m, other=0.0)
        # ci = floor(y) -> [0, nn-2],  f = clamp(y - ci, 0, 1)
        c0 = tl.minimum(tl.maximum(tl.floor(y0).to(tl.int64), 0), n0 - 2)
        c1 = tl.minimum(tl.maximum(tl.floor(y1).to(tl.int64), 0), n1 - 2)
        c2 = tl.minimum(tl.maximum(tl.floor(y2).to(tl.int64), 0), n2 - 2)
        f0 = tl.minimum(tl.maximum(y0 - c0.to(tl.float32), 0.0), 1.0)
        f1 = tl.minimum(tl.maximum(y1 - c1.to(tl.float32), 0.0), 1.0)
        f2 = tl.minimum(tl.maximum(y2 - c2.to(tl.float32), 0.0), 1.0)
        # 내림차순 순위 (같으면 축 번호가 작은 쪽이 앞)
        r0 = (f1 > f0).to(tl.int32) + (f2 > f0).to(tl.int32)
        r1 = (f0 >= f1).to(tl.int32) + (f2 > f1).to(tl.int32)
        r2 = (f0 >= f2).to(tl.int32) + (f1 >= f2).to(tl.int32)
        # 순위 -> 축 (perm)
        p0 = tl.where(r0 == 0, 0, tl.where(r1 == 0, 1, 2))
        p1 = tl.where(r0 == 1, 0, tl.where(r1 == 1, 1, 2))
        p2 = tl.where(r0 == 2, 0, tl.where(r1 == 2, 1, 2))
        s0 = tl.where(p0 == 0, f0, tl.where(p0 == 1, f1, f2))
        s1 = tl.where(p1 == 0, f0, tl.where(p1 == 1, f1, f2))
        s2 = tl.where(p2 == 0, f0, tl.where(p2 == 1, f1, f2))
        tl.store(LAM + p * 4 + 0, 1.0 - s0, mask=m)
        tl.store(LAM + p * 4 + 1, s0 - s1, mask=m)
        tl.store(LAM + p * 4 + 2, s1 - s2, mask=m)
        tl.store(LAM + p * 4 + 3, s2, mask=m)
        tl.store(PERM + p, p0 + 4 * p1 + 16 * p2, mask=m)
        # 사슬 꼭짓점: v0 = ci, 걸음마다 +e_{perm[r]}, v3 = ci + (1,1,1)
        a0 = c0 + (p0 == 0).to(tl.int64)
        a1 = c1 + (p0 == 1).to(tl.int64)
        a2 = c2 + (p0 == 2).to(tl.int64)
        b0 = a0 + (p1 == 0).to(tl.int64)
        b1 = a1 + (p1 == 1).to(tl.int64)
        b2 = a2 + (p1 == 2).to(tl.int64)
        q0 = (c0 * n1 + c1) * n2 + c2
        q1 = (a0 * n1 + a1) * n2 + a2
        q2 = (b0 * n1 + b1) * n2 + b2
        q3 = ((c0 + 1) * n1 + (c1 + 1)) * n2 + (c2 + 1)
        q0 = tl.minimum(tl.maximum(q0, 0), CC - 1)
        q1 = tl.minimum(tl.maximum(q1, 0), CC - 1)
        q2 = tl.minimum(tl.maximum(q2, 0), CC - 1)
        q3 = tl.minimum(tl.maximum(q3, 0), CC - 1)
        tl.store(FLAT + p * 4 + 0, q0.to(tl.int32), mask=m)
        tl.store(FLAT + p * 4 + 1, q1.to(tl.int32), mask=m)
        tl.store(FLAT + p * 4 + 2, q2.to(tl.int32), mask=m)
        tl.store(FLAT + p * 4 + 3, q3.to(tl.int32), mask=m)
        one = tl.full([BLOCK], 1, tl.int32)
        tl.store(MARK + q0, one, mask=m)
        tl.store(MARK + q1, one, mask=m)
        tl.store(MARK + q2, one, mask=m)
        tl.store(MARK + q3, one, mask=m)

    @triton.jit
    def _acc11(G1, r, wm, x0, x1, x2, X0, X1, X2, v0, v1, v2, m):
        base = G1 + r * 11
        tl.atomic_add(base + 0, wm, mask=m, sem="relaxed")
        tl.atomic_add(base + 1, wm * x0, mask=m, sem="relaxed")
        tl.atomic_add(base + 2, wm * x1, mask=m, sem="relaxed")
        tl.atomic_add(base + 3, wm * x2, mask=m, sem="relaxed")
        tl.atomic_add(base + 4, wm * X0, mask=m, sem="relaxed")
        tl.atomic_add(base + 5, wm * X1, mask=m, sem="relaxed")
        tl.atomic_add(base + 6, wm * X2, mask=m, sem="relaxed")
        tl.atomic_add(base + 7, wm * v0, mask=m, sem="relaxed")
        tl.atomic_add(base + 8, wm * v1, mask=m, sem="relaxed")
        tl.atomic_add(base + 9, wm * v2, mask=m, sem="relaxed")
        tl.atomic_add(base + 10, tl.full(wm.shape, 1.0, tl.float32), mask=m, sem="relaxed")

    @triton.jit
    def _edge2(ETAB, ri, rj, c, m):
        one = tl.full(c.shape, 1, tl.int32)
        tl.store(ETAB + ri * 14 + c, one, mask=m)
        tl.store(ETAB + rj * 14 + c + 7, one, mask=m)

    @triton.jit
    def _p2g_first(FLAT, CS, ROWS, LAM, MASS, X, XR, VN, PERM, G1, ETAB, N, MC,
                   BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        p = pid * BLOCK + tl.arange(0, BLOCK)
        m = p < N
        # 압축 색인 = 누적합 - 1 (unique 의 오름차순 색인과 같다)
        r0 = tl.minimum(tl.maximum(tl.load(CS + tl.load(FLAT + p * 4 + 0, mask=m, other=0), mask=m, other=1) - 1, 0), MC)
        r1 = tl.minimum(tl.maximum(tl.load(CS + tl.load(FLAT + p * 4 + 1, mask=m, other=0), mask=m, other=1) - 1, 0), MC)
        r2 = tl.minimum(tl.maximum(tl.load(CS + tl.load(FLAT + p * 4 + 2, mask=m, other=0), mask=m, other=1) - 1, 0), MC)
        r3 = tl.minimum(tl.maximum(tl.load(CS + tl.load(FLAT + p * 4 + 3, mask=m, other=0), mask=m, other=1) - 1, 0), MC)
        tl.store(ROWS + p * 4 + 0, r0, mask=m)
        tl.store(ROWS + p * 4 + 1, r1, mask=m)
        tl.store(ROWS + p * 4 + 2, r2, mask=m)
        tl.store(ROWS + p * 4 + 3, r3, mask=m)
        ms = tl.load(MASS + p, mask=m, other=0.0)
        x0 = tl.load(X + p * 3 + 0, mask=m, other=0.0)
        x1 = tl.load(X + p * 3 + 1, mask=m, other=0.0)
        x2 = tl.load(X + p * 3 + 2, mask=m, other=0.0)
        X0 = tl.load(XR + p * 3 + 0, mask=m, other=0.0)
        X1 = tl.load(XR + p * 3 + 1, mask=m, other=0.0)
        X2 = tl.load(XR + p * 3 + 2, mask=m, other=0.0)
        v0 = tl.load(VN + p * 3 + 0, mask=m, other=0.0)
        v1 = tl.load(VN + p * 3 + 1, mask=m, other=0.0)
        v2 = tl.load(VN + p * 3 + 2, mask=m, other=0.0)
        l0 = tl.load(LAM + p * 4 + 0, mask=m, other=0.0)
        l1 = tl.load(LAM + p * 4 + 1, mask=m, other=0.0)
        l2 = tl.load(LAM + p * 4 + 2, mask=m, other=0.0)
        l3 = tl.load(LAM + p * 4 + 3, mask=m, other=0.0)
        _acc11(G1, r0, l0 * ms, x0, x1, x2, X0, X1, X2, v0, v1, v2, m)
        _acc11(G1, r1, l1 * ms, x0, x1, x2, X0, X1, X2, v0, v1, v2, m)
        _acc11(G1, r2, l2 * ms, x0, x1, x2, X0, X1, X2, v0, v1, v2, m)
        _acc11(G1, r3, l3 * ms, x0, x1, x2, X0, X1, X2, v0, v1, v2, m)
        # 간선표: 사면체 6 변. 변 (i,j) 의 축 비트합 = 걸음 i..j-1 의 축 비트
        pk = tl.load(PERM + p, mask=m, other=0)
        e0 = 1 << (pk & 3)
        e1 = 1 << ((pk >> 2) & 3)
        e2 = 1 << ((pk >> 4) & 3)
        _edge2(ETAB, r0, r1, _cls_of(e0), m)
        _edge2(ETAB, r0, r2, _cls_of(e0 + e1), m)
        _edge2(ETAB, r0, r3, _cls_of(e0 + e1 + e2), m)
        _edge2(ETAB, r1, r2, _cls_of(e1), m)
        _edge2(ETAB, r1, r3, _cls_of(e1 + e2), m)
        _edge2(ETAB, r2, r3, _cls_of(e2), m)

    @triton.jit
    def _acc24(G2, C9, r, wm, x0, x1, x2, X0, X1, X2, v0, v1, v2, m):
        cb = C9 + r * 9
        dx0 = x0 - tl.load(cb + 0, mask=m, other=0.0)
        dx1 = x1 - tl.load(cb + 1, mask=m, other=0.0)
        dx2 = x2 - tl.load(cb + 2, mask=m, other=0.0)
        dX0 = X0 - tl.load(cb + 3, mask=m, other=0.0)
        dX1 = X1 - tl.load(cb + 4, mask=m, other=0.0)
        dX2 = X2 - tl.load(cb + 5, mask=m, other=0.0)
        dv0 = v0 - tl.load(cb + 6, mask=m, other=0.0)
        dv1 = v1 - tl.load(cb + 7, mask=m, other=0.0)
        dv2 = v2 - tl.load(cb + 8, mask=m, other=0.0)
        base = G2 + r * 24
        # dx dx^T 위 삼각 (00 01 02 11 12 22)
        tl.atomic_add(base + 0, wm * (dx0 * dx0), mask=m, sem="relaxed")
        tl.atomic_add(base + 1, wm * (dx0 * dx1), mask=m, sem="relaxed")
        tl.atomic_add(base + 2, wm * (dx0 * dx2), mask=m, sem="relaxed")
        tl.atomic_add(base + 3, wm * (dx1 * dx1), mask=m, sem="relaxed")
        tl.atomic_add(base + 4, wm * (dx1 * dx2), mask=m, sem="relaxed")
        tl.atomic_add(base + 5, wm * (dx2 * dx2), mask=m, sem="relaxed")
        # dx x dv (torch.cross 와 같은 식)
        tl.atomic_add(base + 6, wm * (dx1 * dv2 - dx2 * dv1), mask=m, sem="relaxed")
        tl.atomic_add(base + 7, wm * (dx2 * dv0 - dx0 * dv2), mask=m, sem="relaxed")
        tl.atomic_add(base + 8, wm * (dx0 * dv1 - dx1 * dv0), mask=m, sem="relaxed")
        # dx dX^T (9, 행 우선)
        tl.atomic_add(base + 9, wm * (dx0 * dX0), mask=m, sem="relaxed")
        tl.atomic_add(base + 10, wm * (dx0 * dX1), mask=m, sem="relaxed")
        tl.atomic_add(base + 11, wm * (dx0 * dX2), mask=m, sem="relaxed")
        tl.atomic_add(base + 12, wm * (dx1 * dX0), mask=m, sem="relaxed")
        tl.atomic_add(base + 13, wm * (dx1 * dX1), mask=m, sem="relaxed")
        tl.atomic_add(base + 14, wm * (dx1 * dX2), mask=m, sem="relaxed")
        tl.atomic_add(base + 15, wm * (dx2 * dX0), mask=m, sem="relaxed")
        tl.atomic_add(base + 16, wm * (dx2 * dX1), mask=m, sem="relaxed")
        tl.atomic_add(base + 17, wm * (dx2 * dX2), mask=m, sem="relaxed")
        # dX dX^T 위 삼각
        tl.atomic_add(base + 18, wm * (dX0 * dX0), mask=m, sem="relaxed")
        tl.atomic_add(base + 19, wm * (dX0 * dX1), mask=m, sem="relaxed")
        tl.atomic_add(base + 20, wm * (dX0 * dX2), mask=m, sem="relaxed")
        tl.atomic_add(base + 21, wm * (dX1 * dX1), mask=m, sem="relaxed")
        tl.atomic_add(base + 22, wm * (dX1 * dX2), mask=m, sem="relaxed")
        tl.atomic_add(base + 23, wm * (dX2 * dX2), mask=m, sem="relaxed")

    @triton.jit
    def _p2g_second(ROWS, LAM, MASS, X, XR, VN, C9, G2, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        p = pid * BLOCK + tl.arange(0, BLOCK)
        m = p < N
        ms = tl.load(MASS + p, mask=m, other=0.0)
        x0 = tl.load(X + p * 3 + 0, mask=m, other=0.0)
        x1 = tl.load(X + p * 3 + 1, mask=m, other=0.0)
        x2 = tl.load(X + p * 3 + 2, mask=m, other=0.0)
        X0 = tl.load(XR + p * 3 + 0, mask=m, other=0.0)
        X1 = tl.load(XR + p * 3 + 1, mask=m, other=0.0)
        X2 = tl.load(XR + p * 3 + 2, mask=m, other=0.0)
        v0 = tl.load(VN + p * 3 + 0, mask=m, other=0.0)
        v1 = tl.load(VN + p * 3 + 1, mask=m, other=0.0)
        v2 = tl.load(VN + p * 3 + 2, mask=m, other=0.0)
        for j in tl.static_range(4):
            r = tl.load(ROWS + p * 4 + j, mask=m, other=0)
            wm = tl.load(LAM + p * 4 + j, mask=m, other=0.0) * ms
            _acc24(G2, C9, r, wm, x0, x1, x2, X0, X1, X2, v0, v1, v2, m)

    @triton.jit
    def _g2p(ROWS, LAM, PERM, DP, DPT, X, AI, X2O, V2O, JO, N,
             HAS_T: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        p = pid * BLOCK + tl.arange(0, BLOCK)
        m = p < N
        r0 = tl.load(ROWS + p * 4 + 0, mask=m, other=0)
        r1 = tl.load(ROWS + p * 4 + 1, mask=m, other=0)
        r2 = tl.load(ROWS + p * 4 + 2, mask=m, other=0)
        r3 = tl.load(ROWS + p * 4 + 3, mask=m, other=0)
        l0 = tl.load(LAM + p * 4 + 0, mask=m, other=0.0)
        l1 = tl.load(LAM + p * 4 + 1, mask=m, other=0.0)
        l2 = tl.load(LAM + p * 4 + 2, mask=m, other=0.0)
        l3 = tl.load(LAM + p * 4 + 3, mask=m, other=0.0)
        pk = tl.load(PERM + p, mask=m, other=0)
        q0 = pk & 3
        q1 = (pk >> 2) & 3
        # 축 k 의 순위 (rank = perm 의 역)
        rk0 = tl.where(q0 == 0, 0, tl.where(q1 == 0, 1, 2))
        rk1 = tl.where(q0 == 1, 0, tl.where(q1 == 1, 1, 2))
        rk2 = tl.where(q0 == 2, 0, tl.where(q1 == 2, 1, 2))
        for i in tl.static_range(3):
            u0 = tl.load(DP + r0 * 3 + i, mask=m, other=0.0)
            u1 = tl.load(DP + r1 * 3 + i, mask=m, other=0.0)
            u2 = tl.load(DP + r2 * 3 + i, mask=m, other=0.0)
            u3 = tl.load(DP + r3 * 3 + i, mask=m, other=0.0)
            du = l0 * u0 + l1 * u1 + l2 * u2 + l3 * u3
            xi = tl.load(X + p * 3 + i, mask=m, other=0.0)
            tl.store(X2O + p * 3 + i, xi + du, mask=m)
            # 걸음 r 의 차 d_r = u_{r+1} - u_r ; Dy[i, k] = d_{rank_k}
            d0 = u1 - u0
            d1 = u2 - u1
            d2 = u3 - u2
            D0 = tl.where(rk0 == 0, d0, tl.where(rk0 == 1, d1, d2))
            D1 = tl.where(rk1 == 0, d0, tl.where(rk1 == 1, d1, d2))
            D2 = tl.where(rk2 == 0, d0, tl.where(rk2 == 1, d1, d2))
            for cc in tl.static_range(3):
                g = (D0 * tl.load(AI + 0 * 3 + cc) + D1 * tl.load(AI + 1 * 3 + cc)
                     + D2 * tl.load(AI + 2 * 3 + cc))
                if i == cc:
                    g = g + 1.0
                tl.store(JO + p * 9 + i * 3 + cc, g, mask=m)
            if HAS_T:
                w0 = tl.load(DPT + r0 * 3 + i, mask=m, other=0.0)
                w1 = tl.load(DPT + r1 * 3 + i, mask=m, other=0.0)
                w2 = tl.load(DPT + r2 * 3 + i, mask=m, other=0.0)
                w3 = tl.load(DPT + r3 * 3 + i, mask=m, other=0.0)
                tl.store(V2O + p * 3 + i, l0 * w0 + l1 * w1 + l2 * w2 + l3 * w3, mask=m)

    @triton.jit
    def _edge_msg(HT, P, SRC, DST, CLS, ET, EMB, W1, SN, H: tl.constexpr,
                  W1S: tl.constexpr, HAS_T: tl.constexpr, BE: tl.constexpr,
                  BK: tl.constexpr):
        """간선 메시지 첫 층 + SiLU + dst 로 합 (원값·탄젠트 한 번에).

        z  = W1a (h_src * e_cls) + P_dst          (P = W1b h + b1, 노드에서 미리)
        z' = W1a (t_src * e_cls) + P'_dst
        s  = silu(z),  s' = silu'(z) z'  ->  SN[dst] += (s, s')
        행렬곱은 input_precision="ieee" (TF32 아님, fp32 그대로).
        """
        pid = tl.program_id(0)
        e = pid * BE + tl.arange(0, BE)
        ne = tl.load(ET)
        m = e < ne
        s = tl.load(SRC + e, mask=m, other=0)
        d = tl.load(DST + e, mask=m, other=0)
        c = tl.load(CLS + e, mask=m, other=0)
        n = tl.arange(0, H)
        acc = tl.zeros([BE, H], dtype=tl.float32)
        if HAS_T:
            acct = tl.zeros([BE, H], dtype=tl.float32)
        for k0 in tl.static_range(0, H, BK):
            k = k0 + tl.arange(0, BK)
            em = tl.load(EMB + c[:, None] * H + k[None, :], mask=m[:, None], other=0.0)
            hs = tl.load(HT + s[:, None] * (2 * H) + k[None, :], mask=m[:, None], other=0.0)
            # W1a^T 조각 [BK, H]: W1[n, k]
            w = tl.load(W1 + n[None, :] * W1S + k[:, None])
            acc += tl.dot(hs * em, w, input_precision="ieee")
            if HAS_T:
                ts = tl.load(HT + s[:, None] * (2 * H) + H + k[None, :], mask=m[:, None], other=0.0)
                acct += tl.dot(ts * em, w, input_precision="ieee")
        z = acc + tl.load(P + d[:, None] * (2 * H) + n[None, :], mask=m[:, None], other=0.0)
        sig = 1.0 / (1.0 + tl.exp(-z))
        tl.atomic_add(SN + d[:, None] * (2 * H) + n[None, :], z * sig,
                      mask=m[:, None], sem="relaxed")
        if HAS_T:
            zt = acct + tl.load(P + d[:, None] * (2 * H) + H + n[None, :], mask=m[:, None], other=0.0)
            st = zt * sig * (1.0 + z * (1.0 - sig))
            tl.atomic_add(SN + d[:, None] * (2 * H) + H + n[None, :], st,
                          mask=m[:, None], sem="relaxed")


def _grid(n):
    return (triton.cdiv(n, BLOCK),)


def locate_mark(y, nn, flat, lam, perm, mark):
    n = y.shape[0]
    _locate_mark[_grid(n)](y, nn, flat, lam, perm, mark, n, mark.numel(), BLOCK=BLOCK)


def p2g_first(flat, cs, rows, lam, mass, x, X, vn, perm, g1, etab, mc):
    n = x.shape[0]
    _p2g_first[_grid(n)](flat, cs, rows, lam, mass, x, X, vn, perm, g1, etab, n, mc,
                         BLOCK=BLOCK)


def p2g_second(rows, lam, mass, x, X, vn, c9, g2):
    n = x.shape[0]
    _p2g_second[_grid(n)](rows, lam, mass, x, X, vn, c9, g2, n, BLOCK=BLOCK)


def g2p(rows, lam, perm, dp, dpt, x, Ai, x2, v2, J):
    n = x.shape[0]
    _g2p[_grid(n)](rows, lam, perm, dp, dp if dpt is None else dpt, x, Ai, x2, v2, J, n,
                   HAS_T=dpt is not None, BLOCK=BLOCK)


def edge_msg(ht, P, src, dst, cls, et, emb, W1, sn, has_t, BE=64, BK=32):
    """ht, P, sn: [M, 2, H] 연속. W1: msg 첫 선형층 가중치 [H, 2H] (앞 H 열이 W1a)."""
    H = ht.shape[-1]
    ne_cap = src.shape[0]
    _edge_msg[(triton.cdiv(ne_cap, BE),)](ht, P, src, dst, cls, et, emb, W1, sn, H=H,
                                          W1S=W1.stride(0), HAS_T=has_t, BE=BE, BK=BK,
                                          num_warps=8)
