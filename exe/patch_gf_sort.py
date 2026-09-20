"""GF 러너가 **입자를 격자 칸 순서로 세우고** 돌게 한다. 답은 그대로다.

p2g 가 전체의 70% 인데, 그 커널을 붙잡고 있는 것은 계산량이 아니라 격자 메모리
접근의 지역성이다 (섞으면 4.4 배 느려지고, 칸 순서로 세우면 1.7 배 빨라진다).

입자 순서를 바꾸면 가우시안과의 짝이 어긋나므로, 솔버가 `orig_index` 를 들고
다니고 **내보낼 때와 h5 로 쓸 때 처음 순서로 되돌린다**. 그래서 바깥에서 보는
것은 하나도 안 바뀐다.

  python exe/patch_gf_sort.py --gf <GaussianFluent> [--every 1]
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True)
ap.add_argument("--every", type=int, default=1, help="몇 프레임마다 다시 세울지")
a = ap.parse_args()

MARK = "# [anchorflow] 칸 순서 정렬"

# ---------------------------------------------------- 1) 솔버에 정렬 기능
q = os.path.join(a.gf, "mpm_solver_warp", "mpm_solver_warp.py")
s = open(q).read()
if MARK not in s:
    METHODS = '''
    ''' + MARK + ''': 입자를 격자 칸 순서로 다시 세운다 (물리는 안 바뀐다)
    def af_sort_by_cell(self):
        import torch as _t
        n = self.n_particles
        n_grid = self.mpm_model.n_grid
        dx = self.mpm_model.grid_lim / n_grid
        x = wp.to_torch(self.mpm_state.particle_x)
        c = _t.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) / dx
        c = c.floor().long().clamp_(0, n_grid - 1)
        key = (c[:, 0] * n_grid + c[:, 1]) * n_grid + c[:, 2]
        perm = _t.argsort(key, stable=True)
        if not hasattr(self, "af_orig"):
            self.af_orig = _t.arange(n, device=x.device)
        self.af_orig = self.af_orig[perm].contiguous()
        for obj in (self.mpm_state, self.mpm_model):
            for nm in dir(obj):
                if nm.startswith("_"):
                    continue
                arr = getattr(obj, nm, None)
                if not isinstance(arr, wp.array) or arr.shape[0] not in (n, 6 * n):
                    continue
                t = wp.to_torch(arr)
                if t.shape[0] == n:
                    t.copy_(t[perm].contiguous())
                else:
                    t.copy_(t.view(n, 6)[perm].reshape(-1).contiguous())

    def af_unsort(self, t):
        """지금 순서로 된 것을 **처음 순서**로 되돌린다."""
        if not hasattr(self, "af_orig"):
            return t
        import torch as _t
        if not hasattr(self, "af_inv") or self.af_inv_tag is not id(self.af_orig):
            self.af_inv = _t.argsort(self.af_orig)
            self.af_inv_tag = id(self.af_orig)
        return t[self.af_inv]
'''
    anchor = "    def export_particle_x_to_torch(self):"
    if anchor not in s:
        raise SystemExit("export_particle_x_to_torch 를 못 찾았다")
    s = s.replace(anchor, METHODS + "\n" + anchor, 1)
    # 내보내기는 처음 순서로 돌려준다
    for nm, fld in (("x", "particle_x"), ("v", "particle_v"), ("F", "particle_F"),
                    ("C", "particle_C"), ("R", "particle_R"), ("cov", "particle_cov")):
        old = f"    def export_particle_{nm}_to_torch(self):\n        return wp.to_torch(self.mpm_state.{fld})"
        new = (f"    def export_particle_{nm}_to_torch(self):\n"
               f"        return self.af_unsort(wp.to_torch(self.mpm_state.{fld}))")
        s = s.replace(old, new)
    open(q, "w").write(s)
    print(f"고쳤다: {q}")

# ---------------------------------------------------- 2) h5 도 처음 순서로
e = os.path.join(a.gf, "mpm_solver_warp", "engine_utils.py")
s = open(e).read()
if MARK not in s:
    OLD = """    if save_to_h5:

        if os.path.exists(fullfilename):"""
    NEW = ("    if save_to_h5:\n\n        " + MARK +
           ": 저장은 항상 처음 순서로 한다\n"
           "        _inv = None\n"
           "        if hasattr(mpm_solver, 'af_orig'):\n"
           "            import torch as _t\n"
           "            _inv = _t.argsort(mpm_solver.af_orig).cpu().numpy()\n"
           "        def _u(v):\n"
           "            return v if _inv is None else v[_inv]\n\n"
           "        if os.path.exists(fullfilename):")
    if OLD not in s:
        raise SystemExit("save_data_at_frame 의 h5 갈래를 못 찾았다")
    s = s.replace(OLD, NEW, 1)
    for fld in ("particle_x", "particle_F", "particle_v", "particle_C",
                "particle_cov", "particle_selection", "particle_Jp"):
        s = s.replace(f"mpm_solver.mpm_state.{fld}.numpy()",
                      f"_u(mpm_solver.mpm_state.{fld}.numpy())")
    open(e, "w").write(s)
    print(f"고쳤다: {e}")

# ---------------------------------------------------- 3) 러너가 부르게 한다
for rel in ("gs_simulation.py",
            os.path.join("gs_simulation", "watermelon", "gs_simulation_watermelon.py")):
    f = os.path.join(a.gf, rel)
    if not os.path.exists(f):
        continue
    s = open(f).read()
    if MARK in s:
        continue
    OLD = "    for frame in tqdm(range(frame_num)):"
    if OLD not in s:
        print(f"[건너뜀] {rel}: 프레임 루프를 못 찾았다")
        continue
    NEW = ("    " + MARK + f": 처음 한 번, 그리고 {a.every} 프레임마다\n"
           "    mpm_solver.af_sort_by_cell()\n" + OLD +
           f"\n        if frame % {a.every} == 0:\n"
           "            mpm_solver.af_sort_by_cell()")
    s = s.replace(OLD, NEW, 1)
    open(f, "w").write(s)
    print(f"고쳤다: {f}")
print("GFSORT_OK")
