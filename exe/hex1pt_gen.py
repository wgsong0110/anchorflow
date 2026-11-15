#!/usr/bin/env python3
"""절 CM179: 육면체 셀 (이차 B-스플라인) 가우스 적분 2x2x2 (기본) 대 1x1x1 (칸 가운데 1 점) 명령 생성.
사용: hex1pt_gen.py <cm157 의 A 판 cmd.txt (R2 lbours/wolfcv/cm157/{LA_1,FA_1,BA_1})> <L|F|B> <2|1> <출력 디렉토리>  -> 표준출력에 명령줄.
공통 기본 판 명령 그대로에서 바꾸는 것만: 프레임 수 (레고·ficus 50, bread 원래 길이 300) + AF_SP_FR, 덤프 (--dump --dump_F --dump_part; 영상·모드 분석용, 결과 불변),
q27 감싸개 (2 = startpt 바로 뒤, 1 = 글자 고침은 실행기 바로 앞 (가장 안쪽) + 클래스 고침은 startpt 바로 뒤 (가장 바깥); 2 = `--q27 0 --q27dg 1` 모듈은 풀이기 정보만 저장, 결과 불변 / 1 = `--q1pt 1 --q27dg 1` 칸 가운데 1 점)."""
import re, sys
src, scene, mode, out = sys.argv[1:5]
c = " ".join(open(src).read().split())
fr = int(re.search(r"--frames (\d+)", c).group(1))
if scene in "LF":
    nf = 50
    c = c.replace(f"--frames {fr}", f"--frames {nf}")
    c = c.replace(f"AF_SP_FR=1,{fr} ", f"AF_SP_FR=1,{nf} ")
    assert f"AF_SP_FR=1,{nf} " in c
assert c.count("exe/startpt_wrap.py --sp 1 ") == 1
if mode == "1":                                       # 1x1x1: 가장 안쪽 감싸개 (실행기 바로 앞) -- 최종 CUDA 글자 (finterp 등 고침 뒤) 를 고치려고
    assert c.count(" exe/fem_an_rt_sim.py ") == 1
    c = c.replace(" exe/fem_an_rt_sim.py ", " exe/q27_wrap.py --q1pt 1 --q27dg 1 --q27ph src exe/fem_an_rt_sim.py ")   # 글자 고침 (가장 안쪽)
    c = c.replace("exe/startpt_wrap.py --sp 1 ", "exe/startpt_wrap.py --sp 1 exe/q27_wrap.py --q1pt 1 --q27dg 1 --q27ph cls ")   # 클래스 고침·점검 (가장 바깥)
else:
    c = c.replace("exe/startpt_wrap.py --sp 1 ", "exe/startpt_wrap.py --sp 1 exe/q27_wrap.py --q27 0 --q27dg 1 ")
c = c.replace(" --dump ", " ")
assert " --K 1 " in c
c = c.replace(" --K 1 ", " --K 1 --dump --dump_F --dump_part ", 1)
c = re.sub(r"--out \S+$", f"--out {out}", c)
print(c)
