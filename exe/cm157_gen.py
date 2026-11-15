#!/usr/bin/env python3
"""절 CM157: 레고(L)·ficus(F)·bread(B) 의 H9 판 (A) 과 H9 − 확장 − S_구속·C + --mlow 1e-20 판 (B) 명령 생성.
사용: cm157_gen.py <H7 명령 파일 (R2 lbours/ext2/oA{L,F,B}_1/cmd.txt)> <L|F|B> <A|B> <출력 디렉토리>  -> 표준출력에 명령줄.
기준 = 절 CM84 H7 회귀 명령 (ext2 oA*_1) 에서: 렌더·예산 인자 (--rr*, --tc 8, --s4*) 는 H7 회귀 명령 그대로 두고 (bread 정답 궤적 경로만 인스턴스 자리로), 감싸개를 절 CM139 H9 사슬로 바꿈
(startpt --sp 1 · nlwhy · wolfh12 --w12 0 --w12dg 1 · finterp --fi 0 --fidg 1 · wolfh11 · wolfh9 · speed12 --s12z 1 · ext2). 프레임 수는 그대로."""
import re, sys
src, scene, kind, out = sys.argv[1:5]
c = " ".join(open(src).read().split())
fr = int(re.search(r"--frames (\d+)", c).group(1))
c = c.replace("/dev/shm/dkta/t2chain/gt_bread", "/dev/shm/dkta/ag_data/gt_bread")                  # bread 정답 궤적 (렌더 카메라·0 프레임) 자리
H9 = ("exe/startpt_wrap.py --sp 1 exe/nlwhy_wrap.py exe/wolfh12_wrap.py --w12 0 --w12dg 1 exe/finterp_wrap.py --fi 0 --fidg 1 "
      "exe/wolfh11_wrap.py exe/wolfh9_wrap.py exe/speed12_wrap.py --s12z 1 exe/ext2_wrap.py")
assert c.count("exe/ext2_wrap.py") == 1
c = c.replace("exe/ext2_wrap.py", H9)
c = f"AF_SP_FR=1,{fr} " + c.strip()
if kind == "B":
    for t in (" --w8e 1 --w8d 2 --w8t 1", " --w5s 1 --w5c 1"):
        assert t in c, t
        c = c.replace(t, "")
    if "--mlow " in c:
        c = re.sub(r"--mlow \S+", "--mlow 1e-20", c)
    else:
        c = c.replace(" --cell 1 ", " --mlow 1e-20 --cell 1 ", 1)
if __import__("os").environ.get("CM157_HFF", "0") == "1":   # 절 CM184 검증 (기본 0 = 기존 그대로): hexff 감싸개 (exe/hexff_wrap.py --hff 1) 를 nlwhy 바로 앞에 (exe/gfs_cmd.py 와 같은 자리)
    assert c.count("exe/nlwhy_wrap.py") == 1
    c = c.replace("exe/nlwhy_wrap.py", "exe/hexff_wrap.py --hff 1 exe/nlwhy_wrap.py")
c = re.sub(r"--out \S+$", f"--out {out}", c)
print(c)
