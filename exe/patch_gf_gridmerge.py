"""격자 세 배열을 **한 칸 32 바이트 안에 겹쳐 놓는다**. 커널 코드는 안 고친다.

왜. p2g 는 노드마다 `grid_m`(4B), `grid_v_in`(12B), `grid_v_out`(12B) 를 건드리는데
셋이 따로 떨어져 있어 서로 다른 32 바이트 섹터 **세 개**를 끌어온다. 쓸모 있는
바이트는 28 인데 96 을 실어 나르는 셈이다. 실측 유효 대역폭이 835 GB/s 로 이미
3090 의 한계에 붙어 있었다 -- 계산이 아니라 이 낭비가 p2g 를 붙잡고 있다.

한 칸을 [m, v_in(3), v_out(3), 여분] 32 바이트로 묶고, 세 이름을 그 버퍼를
가리키는 **보폭 배열**로 다시 매어 둔다. 커널에서 보는 이름과 뜻이 그대로라
`mpm_utils.py` 는 한 줄도 안 바뀐다.

  python exe/patch_gf_gridmerge.py --gf <GaussianFluent>
"""
import argparse
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
a = ap.parse_args()

MARK = "# [anchorflow] 격자 세 배열을 한 칸에 겹친다"
q = os.path.join(a.gf, "mpm_solver_warp", "mpm_solver_warp.py")
s = open(q).read()
if MARK in s:
    print("이미 되어 있다"); raise SystemExit(0)

def _span(text, i):
    """`... = wp.zeros(` 한 문장의 끝(닫는 괄호 다음)을 찾는다."""
    j = text.index("(", i)
    d = 0
    while True:
        if text[j] == "(":
            d += 1
        elif text[j] == ")":
            d -= 1
            if d == 0:
                return j + 1
        j += 1


NEW = '''{i}{mark}
{i}_n = self.mpm_model.n_grid
{i}self.af_grid_buf = wp.zeros(shape=(_n, _n, _n, 8), dtype=float, device=device)
{i}_p = self.af_grid_buf.ptr
{i}_cell = 8 * 4                      # 한 칸 32 바이트 (7 개 쓰고 1 개는 여분)
{i}_st = (_n * _n * _cell, _n * _cell, _cell)
{i}self.mpm_state.grid_m = wp.array(
{i}    ptr=_p + 0, dtype=float, shape=(_n, _n, _n), strides=_st,
{i}    device=device, owner=False)
{i}self.mpm_state.grid_v_in = wp.array(
{i}    ptr=_p + 4, dtype=wp.vec3, shape=(_n, _n, _n), strides=_st,
{i}    device=device, owner=False)
{i}self.mpm_state.grid_v_out = wp.array(
{i}    ptr=_p + 16, dtype=wp.vec3, shape=(_n, _n, _n), strides=_st,
{i}    device=device, owner=False)'''

spans = []
pos = 0
while True:
    k = s.find("self.mpm_state.grid_m = wp.zeros(", pos)
    if k < 0:
        break
    ls = s.rfind("\n", 0, k) + 1
    ind = s[ls:k]
    k2 = s.index("self.mpm_state.grid_v_out = wp.zeros(", k)
    end = _span(s, k2)
    spans.append((ls, end, ind))
    pos = end
if len(spans) != 2:
    raise SystemExit(f"격자 배열 할당 두 곳을 찾아야 하는데 {len(spans)} 곳이다")
for ls, end, ind in reversed(spans):
    s = s[:ls] + NEW.format(i=ind, mark=MARK) + "\n" + s[end:]

open(q, "w").write(s)
print(f"고쳤다: {q} ({len(spans)} 곳)")
print("GFMERGE_OK")
