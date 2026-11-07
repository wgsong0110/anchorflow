"""절 CM37 묶음 common3 (조정자 위임 11-08): 공통 후보 C = BG + '무너진 걸음만 쪼개기' + BG 시간 증가 고침 (결과 불변).

(1) --ks_big (두 실행기, --ksplit D 위; 끄면 split_step 글자 그대로의 판정): 쪼개기 판정을 '덜 풀린 걸음 중 걸음 끝 잔차 비 ‖g‖²/F² (판정 2 값,
    마지막으로 돈 뉴턴 반복 시작) 가 --ue_bigT (1e6) 를 넘거나 비유한' 인 걸음이 있을 때만으로 -- --ue_loc 비트 32 와 같은 판정 (ul_big: !(g2 <= T)).
    쪼개기 경로 (입자 상태 되돌림, K 두 배, 최대 2^D) 는 절 CM24 split_step 그대로.
(2) --xg 1 (두 실행기, --xit_from j 위; 끄면 그래프·소스 그대로): 한 프레임 그래프를 걸음마다 세 경계 (반복 j 시작 'q', 반복 j 의 덜 풀림 판정 뒤 'g',
    뉴턴 고리 끝 'e') 에서 나눠 캡처하고, 'q'·'g' 뒤 구간은 그 앞 구간 끝에서 CT[0] (이 걸음 뉴턴 끝 표시) 를 호스트가 읽어 0 일 때만 재생.
    CT[0] = 1 이면 그 구간 커널은 모두 AF_SKIP 으로 바로 돌아가고 (rt_ctl 은 CT[1] = 0·AF_SKIP 만 -- 걸음 끝 op 8 이 같은 값으로 덮음)
    그래프 안 복사 (진단 DG·선탐색 방향 p0·en_mo y 사본) 는 다음 사용 전에 다시 쓰이므로 결과 (입자 상태) 는 비트 같음.
    --xg 2 = 프로파일: 모든 구간을 늘 재생 (원래 그래프와 같은 일) 하며 '건너뛸 구간' 과 '도는 구간' 의 GPU 시간을 따로 잼.
"""
import torch


# ---------------- (1) 무너진 걸음 판정 ----------------
def nfail(s, fl, g2, K):
    """split_step 의 '이 시도에 다시 할 걸음 수'. s.ksbig > 0 이면 덜 풀린 걸음 ∧ !(‖g‖²/F² <= T), 아니면 덜 풀린 걸음 수 (원래)."""
    T = float(getattr(s, "ksbig", 0.0) or 0.0)
    if T <= 0.0:
        return int(fl[:K].sum())
    return int(((fl[:K] > 0) & ~(g2[:K] <= T)).sum())


# ---------------- (2) 나눈 그래프 ----------------
def mark(s, kind):
    """record() 안 구간 경계. 나눈 캡처 중이 아니면 아무것도 안 함 (커널·복사 없음 -> 그래프 그대로)."""
    c = getattr(s, "_c3cap", None)
    if c is None:
        return
    c["cur"].capture_end()
    c["segs"].append((c["cond"], c["cur"]))
    if c["pool"] is None:
        c["pool"] = c["segs"][0][1].pool()
    g = torch.cuda.CUDAGraph()
    g.capture_begin(pool=c["pool"])
    c["cur"], c["cond"] = g, kind in ("q", "g")


class SegGraph:
    """나눈 그래프 묶음: replay() 가 조건 구간을 CT[0] 로 고름 (step() 은 s.graph.replay() 를 그대로 부름)."""

    def __init__(self, s, segs):
        self.s, self.segs = s, segs

    def replay(self):
        s = self.s
        mode = int(s.xg)
        st = s.c3st
        skip = False
        for cond, g in self.segs:
            if not cond:
                skip = False
                if mode == 2:
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record(); g.replay(); e1.record(); st["ev"].append((0, e0, e1))
                else:
                    g.replay()
                continue
            if not skip:                                                               # 한 번 건너뛰면 걸음 끝까지 CT[0] = 1 그대로 (op 0·1 만 지움)
                skip = bool(int(s.CT[0]) != 0)
            st["sk" if skip else "run"] += 1
            if mode == 1 and skip:
                continue
            if mode == 2:
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); g.replay(); e1.record(); st["ev"].append((2 if skip else 1, e0, e1))
            else:
                g.replay()


def stats_reset(s):
    s.c3st = dict(sk=0, run=0, ev=[])


def frame_stats(s):
    """이 프레임 (모든 시도) 조건 구간 수 [건너뜀, 돎]; --xg 2 면 GPU ms [늘 도는 구간, 조건 구간 중 돈 것, 조건 구간 중 건너뛸 것]."""
    st = s.c3st
    o = dict(sk=st["sk"], run=st["run"])
    if st["ev"]:
        torch.cuda.synchronize()
        m = [0.0, 0.0, 0.0]
        for k_, e0, e1 in st["ev"]:
            m[k_] += e0.elapsed_time(e1)
        o["ms"] = [round(q, 4) for q in m]
    stats_reset(s)
    return o


def capture(s):
    """RTSolver.capture 와 같은 데움 + 나눈 캡처 (같은 메모리 풀). 지속형 판 (--pk) 은 안 됨."""
    assert not getattr(s, "pk", False), "--xg 는 그래프 판만"
    s.mod.get_function("rt_reset")
    K = s.Kcur
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        s.record(0.0, K)                                                               # 데움 (함수 적재)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    st2 = torch.cuda.Stream()
    g0 = torch.cuda.CUDAGraph()
    with torch.cuda.stream(st2):
        g0.capture_begin()
        c = s._c3cap = dict(cur=g0, cond=False, segs=[], pool=None)
        try:
            s.record(0.0, K)
        finally:
            s._c3cap = None
        c["cur"].capture_end()
        c["segs"].append((c["cond"], c["cur"]))
    torch.cuda.synchronize()
    g = SegGraph(s, c["segs"])
    s.graph = g
    s.graphs[K] = g


def install(s, mode):
    """--xg mode (1 고름, 2 프로파일) 를 이 풀이기에 (인스턴스 capture 를 바꿈; 그래프는 다음 캡처부터)."""
    import types
    s.xg = int(mode)
    stats_reset(s)
    s.capture = types.MethodType(capture, s)
    s.graph = None
    if hasattr(s, "graphs"):
        s.graphs.clear()
