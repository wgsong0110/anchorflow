"""절 CM61 묶음 hhtc2 감싸개 (exe/hhtc_wrap.py 복사, 모듈만 lib/anchorflow/fem_hhtc2.py): 충돌 걸음 HHT 갱신 정의 (--hc M, 0~4 절 CM58 그대로 + 진단 hc_mom, 5 H5 · 6 H6) 를 단 뒤 exe/ghost6_wrap.py 로 넘김 (그쪽 인자 그대로).

사용: python exe/hhtc_wrap.py [--hc M] [--tc M (절 CM56, 8 = Rf)] [--rr... (rtrender1 인자)] [--s4... (speed4 인자)] <ghost6_wrap 인자...> <실행기.py> <실행기 인자...>
  --hc M   0 진단만 (결과 불변) · 1 H1 충돌 걸음 몸 전체 후퇴 오일러 · 2 H2 새 접촉 노드 후퇴 오일러 · 3 H3 걸음 끝 접촉 노드 가속도 기억 0. 없으면 모듈을 안 부름.
  --rr*    있으면 ghost6_wrap 이 escape4_wrap 을 띄우는 자리에 exe/rtrender1_wrap.py (렌더 포함 단독 시간, 절 CM54) -> speed4_wrap (--s4* 인자, 절 CM50 A3) 를 끼움.
모듈 설치는 실행기 (fem_an_rt_sim.py / fem_ao_rt_sim.py) 를 runpy 로 띄우는 순간 (다른 감싸개가 소스·k 를 다 감싼 뒤 = 가장 바깥).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
os.makedirs(out, exist_ok=True)
HC = None
TCM = None
RR, S4 = [], []
while argv and (argv[0].startswith("--hc") or argv[0].startswith("--rr") or argv[0].startswith("--s4") or argv[0] == "--tc"):
    k_, v_ = argv[0], argv[1]
    del argv[:2]
    if k_ == "--hc":
        HC = int(float(v_))
    elif k_ == "--tc":
        TCM = int(float(v_))
    elif k_.startswith("--rr"):
        RR += [k_, v_]
    else:
        S4 += [k_, v_]
if TCM is not None:                                                                    # 절 CM56 시간 예산 정의 (조정자 11-08: Rf = --tc 8 공통 채택) -- speed3 tb_install 보다 먼저
    assert "--tb" in argv and (RR or not (TCM & 8)), "--tc 는 --tb 판, 비트 8 (앞 프레임 렌더 빼기) 은 렌더 판 (--rr) 위에서만"
    from anchorflow import fem_t2chain as TC
    TC.install(TCM, out)
assert not S4 or RR, "--s4* 는 --rr (렌더 판) 과 함께만 (speed4_wrap 을 끼우는 자리)"

_orig = runpy.run_path
_st = dict(rr=False, hc=False)


def run_path(path, *a, **kw):
    b = os.path.basename(path)
    if RR and not _st["rr"] and b == "escape4_wrap.py":
        _st["rr"] = True
        p = os.path.join(HERE, "rtrender1_wrap.py")
        sys.argv = [p] + RR + S4 + sys.argv[1:]
        print(f"[절 CM58] 렌더 판: ghost6_wrap -> rtrender1_wrap {' '.join(RR)} -> speed4_wrap {' '.join(S4)} -> escape4_wrap", flush=True)
        return _orig(p, *a, **kw)
    if HC is not None and not _st["hc"] and b in ("fem_an_rt_sim.py", "fem_ao_rt_sim.py"):
        _st["hc"] = True
        from anchorflow import fem_hhtc2 as HCM
        HCM.install(HC, out)
    return _orig(path, *a, **kw)


runpy.run_path = run_path
sys.argv = [os.path.join(HERE, "ghost6_wrap.py")] + argv
_orig(sys.argv[0], run_name="__main__")
