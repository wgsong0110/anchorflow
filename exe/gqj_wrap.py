"""절 CM194 gq_justify 감싸개 (가장 바깥): lib/anchorflow/fem_gq1.py 를 단 뒤 다음 감싸개로 넘김 (그쪽 인자 그대로).
사용: python exe/gqj_wrap.py [--gq_q1 0|1] [--gq_hg γ] [--gq_hgm 0|1] [--gq_vs s] [--gq_ck 0|1] <다음 감싸개.py> <그 인자...>
  --gq_q1 1   칸당 1x1x1 (칸 가운데 1 점; 기본 판 8 레인 배치 그대로)       --gq_hg γ   모래시계 억제 벌점 γμ (1 점 위에서만)
  --gq_hgm 1  모래시계 에너지 비 기록 (hgm.jsonl, 결과 불변·그래프 밖)        --gq_vs s   충격 x s (중력 x s², 경계 속도 x s)
  --gq_ck 1   벌점 커널 참값 검사 (gq1ck.json, 결과 불변)
모두 기본값이면 아무것도 안 함."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
argv = sys.argv[1:]
out = argv[argv.index("--out") + 1] if "--out" in argv else "."
O = dict(gq_q1="0", gq_hg="0", gq_hgm="0", gq_vs="1", gq_ck="0")
while argv and argv[0][2:] in O:
    O[argv[0][2:]] = argv[1]
    del argv[:2]
if any(float(O[k]) != (1.0 if k == "gq_vs" else 0.0) for k in O):
    from anchorflow import fem_gq1 as GQ
    GQ.install(out, q1=int(float(O["gq_q1"])), hg=float(O["gq_hg"]), hgm=int(float(O["gq_hgm"])), vs=float(O["gq_vs"]), ck=int(float(O["gq_ck"])))
sys.argv = list(argv)
runpy.run_path(sys.argv[0], run_name="__main__")
