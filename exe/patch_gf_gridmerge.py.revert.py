import argparse, os, re
ap = argparse.ArgumentParser(); ap.add_argument("--gf", required=True); a = ap.parse_args()
q = os.path.join(a.gf, "mpm_solver_warp", "mpm_solver_warp.py")
s = open(q).read()
MARK = "# [anchorflow] 격자 세 배열을 한 칸에 겹친다"
ORIG = '''{i}self.mpm_state.grid_m = wp.zeros(
{i}    shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
{i}    dtype=float, device=device,
{i})
{i}self.mpm_state.grid_v_in = wp.zeros(
{i}    shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
{i}    dtype=wp.vec3, device=device,
{i})
{i}self.mpm_state.grid_v_out = wp.zeros(
{i}    shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
{i}    dtype=wp.vec3, device=device,
{i})'''
n = 0
while MARK in s:
    k = s.index(MARK)
    ls = s.rfind("\n", 0, k) + 1
    ind = s[ls:k]
    end = s.index("device=device, owner=False)", s.index("grid_v_out = wp.array", k))
    end = s.index("\n", end) + 1
    s = s[:ls] + ORIG.format(i=ind) + "\n" + s[end:]
    n += 1
open(q, "w").write(s)
print(f"되돌렸다 {n} 곳"); print("REVERT_OK")
