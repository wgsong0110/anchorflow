# FusedGS — code to reproduce the paper results

FusedGS simulates 3D Gaussian Splatting scenes in real time. At every frame it rebuilds a hexahedral mesh from the current particles (Gaussians), advances the mesh by one implicit FEM step (2×2×2 Gauss points per cell), and maps the result back to the particles. Because the mesh is rebuilt every frame, topological changes such as tearing need no special handling.

This folder contains only the code that produced the paper's numbers. The FusedGS solver is a set of modifications that were layered on top of a base solver during development. `exe/run_fusedgs.py` applies exactly the modifications and flags of the recorded paper runs, so every paper setting is selected by a name.

## 1. Layout

```
FusedGS_code/
  README.md
  cfg/settings.json        paper settings: scene -> setting -> frames + solver arguments
  cfg/nshape/*.json        PhysGaussian configs of the NeRF-synthetic scenes (ship, mic, drums, chair, ...)
  exe/run_fusedgs.py       FusedGS entry point: expands a setting and installs the lib/ components in the recorded order
  exe/evaluation.py        stability analysis, physics / visual metrics, 3DGS rendering, zero-energy modes
  exe/expressiveness.py    tracking targets and the tracking comparison (expressiveness table)
  exe/tools.py             data conversion (PhysGaussian .h5 -> FusedGS input), new-shape configs, substep-stability driver
  exe/baseline_mpm.py      PhysGaussian, i-PhysGaussian, OmniPhysGS (stability and speed-accuracy runners, i-PG source patches, ground-truth timing)
  exe/baseline_spring.py   PhysTwin and Spring-Gaus on top of their official code
  exe/baseline_vrgs.py     VR-GS (XPBD on a tetrahedral cage; no public code, our implementation -- see its docstring)
  lib/hexfem/               FEM core: per-frame hexahedral grid, implicit FEM step (Newton + multigrid PCG), materials
                            (elastic, plasticine, sand, foam), B-spline/particle transfer; simulate.py = per-scene simulation loop
  lib/quadrature/           Gauss-point rules (1^3, 3^3, 4^3) and deformation-gradient interpolation (F transfer)
  lib/stability/            substep control and robustness: contact barrier, ghost-penalty / light-cell terms, escape and
                            retry logic, rotating boundary conditions, stability statistics
  lib/realtime/             real-time execution: per-frame time budget, CUDA graphs, fused kernels
  lib/gsrender/             3DGS side: in-loop renderer, Gaussian deformation/skinning helpers used by evaluation and tracking
  lib/xpbd/                 XPBD kernels of the VR-GS baseline
                            every package = __init__.py + <package>.cu (CUDA sources, compiled at run time by cupy)
```

Run every command from the `FusedGS_code` folder. Each script prints its subcommands and options with `--help`. Comments in the source are partly in Korean.

## 2. Environment

- Linux, CUDA 12.x, one NVIDIA GPU. All timings in the paper were measured on one RTX A6000 with no other process on the card.
- Python 3.10/3.11, PyTorch 2.5.1 (CUDA 12.1), then:
  ```bash
  pip install cupy-cuda12x taichi==1.6.0 warp-lang==1.6.2 kaolin==0.18.0 numpy scipy h5py plyfile tqdm imageio pyiqa==0.1.16 kornia einops cma ftfy regex
  pip install --no-deps git+https://github.com/openai/CLIP.git   # CLIP-IQA in evaluation.py pareto-vis
  ```
- 3DGS rasterizer: `diff_gaussian_rasterization` and `simple_knn` from the official 3D Gaussian Splatting repository.
- PhysTwin (`exe/baseline_spring.py phystwin`) needs open3d and warp-lang 1.x (we used warp-lang 1.4.2, open3d 0.18.0, numpy 1.26.4; the paper runs used warp-lang 1.9.1, open3d 0.19.0, numpy 2.4). open3d older than 0.19 crashes with numpy 2 in the spring search; the script stops with a message in that case.
- FusedGS itself needs no compilation.
- External code goes into one work folder, given by the environment variable `FUSEDGS_WORK`:
  ```
  $FUSEDGS_WORK/PhysGaussian       official PhysGaussian (ground truth, scene configs, 3DGS loader)
  $FUSEDGS_WORK/i-physgaussian     public i-PhysGaussian re-implementation (patched by exe/baseline_mpm.py patch-*)
  $FUSEDGS_WORK/OmniPhysGS         official OmniPhysGS
  $FUSEDGS_WORK/PhysTwin           official PhysTwin
  $FUSEDGS_WORK/Spring-Gaus        official Spring-Gaus
  $FUSEDGS_WORK/pgmodel/<model>    trained 3DGS models: lego_whitebg-trained, pillow2sofa_whitebg-trained, bread-trained,
                                   wolf_whitebg-trained (PhysGaussian), ship, mic, drums, chair (NeRF-synthetic, 3DGS 30k iterations)
  $FUSEDGS_WORK/stab/<scene>/      stability-table scenes: init/sim_0000000000.h5, config.json, rr/ (render trajectory)
  ```

## 3. Scene data

Every scene has a ground-truth trajectory from PhysGaussian MPM (official code, substep of the scene config; Lego 2.0008e-5 s = 833 substeps per frame). Convert it once into the FusedGS input and a trajectory folder:

```bash
export FUSEDGS_WORK=/path/to/work
export PYTHONUTF8=1      # logs and json files contain non-ASCII text
D=/path/to/data          # scene inputs
O=/path/to/outputs       # results
cd FusedGS_code
python exe/tools.py pg-h5-to-pt --h5_dir <PhysGaussian output>/simulation_ply --config <scene config>.json --out_pt $D/<scene>.pt --out_dir $D/gt_<scene>
```

This gives `$D/<scene>.pt` and `$D/gt_<scene>/` (`sim_*.h5`, `meta.json`, `config.json`). The scenes are lego, pillow2sofa, bread, and wolf (PhysGaussian assets; bread uses the plasticine config of the tearing scene), plus ship, mic, drums, and chair. The configs of the new shapes come from `python exe/tools.py plastfix-lego-cfg`, which writes `cfg/nshape/<scene>_lego.json`. The pillow2sofa config refers to its PhysGaussian `.h5` folder for the boundary conditions; keep that path valid.

## 4. Reproducing each result

FusedGS runs write their output folder: the log on stdout, per-frame error and timing, and for most settings `pred.pt` (particle positions and deformation gradients per frame). The error printed at the end ("평균 …%" = mean over frames) is the error of the speed–accuracy figure. The solver uses a 16.7 ms per-frame time budget (real-time setting), so two runs of the same setting differ slightly (about ±3 % relative on the mean error). Use `--extra '--tb 1e5'` to remove the budget.

### 4.1 Speed–accuracy (Fig. Pareto) and tearing (Fig. bread)

FusedGS, 3 repeats per scene, median:

```bash
for s in lego pillow2sofa bread wolf; do for r in 1 2 3; do
  python exe/run_fusedgs.py --scene $s --setting default --data $D/$s.pt --gt $D/gt_$s --out $O/ours_${s}_r$r
done; done
```

Physics and visual metrics of one run (`--floor vel` for lego and wolf, `none` for pillow2sofa and bread):

```bash
s=bread; F=none; R=$O/ours_${s}_r1
python exe/evaluation.py phys --method ours --scene $s --N default --data $D/$s.pt --pred $R/pred.pt --floor $F --out $R/phys.json
python exe/evaluation.py roll-to-h5 --dump $R/pred.pt --traj $D/gt_$s --out $R/traj
python exe/evaluation.py render --sim $R/traj --config $D/gt_$s/config.json --shape $s --png $R/png --out $R/$s.mp4
mkdir -p $O/gt_$s
python exe/evaluation.py render --sim $D/gt_$s --config $D/gt_$s/config.json --shape $s --png $O/gt_$s/png --out $O/gt_$s/$s.mp4
python exe/evaluation.py pareto-vis --png $R/png --ref $O/gt_$s/png --out $R/vis.json
```

The tearing figure shows frames 0, 30, …, 300 of these bread renders, with the same renders of the baselines below.

Baselines. Scene flags: lego `--floor vel`, pillow2sofa and bread `--floor none --cfg_bc --pg_vdamp`, wolf `--floor vel --cfg_bc`. Frames: lego 80, pillow2sofa 300, bread 300, wolf 50. PG below means the PhysGaussian `.h5` folder of the scene.

| Method | Command (one scene) | Paper cells |
|---|---|---|
| Ground-truth MPM timing | `python exe/baseline_mpm.py gt-timing --gt PG --pt $D/$s.pt --config CFG --nsub 833 --time --out $O/mpm_$s` | substeps of the config |
| i-PhysGaussian | `python exe/baseline_mpm.py patch-<name> --ipg $FUSEDGS_WORK/i-physgaussian` (all 8 patches, once), then `python exe/baseline_mpm.py pareto --method ipg --nsub N --gt PG --config CFG --pt $D/$s.pt --frames T --pg_cfg --time --dump --out $O/ipg_$s` | N = 42 / 10 / 10 / 1000 (lego / pillow2sofa / bread / wolf). Prefix `AF_IPG_FUSED=1` for the elastic scenes lego and pillow2sofa (fused kernel, same numbers, ~20x faster); bread and wolf run with `AF_IPG_FUSED=0`. lego is run without `--pg_cfg`. |
| OmniPhysGS | `python exe/baseline_mpm.py pareto --method omni --nsub N --gt PG --config CFG --pt $D/$s.pt --frames T --pg_cfg --time --dump --out $O/omni_$s` | N = 56 / 67 / 34 (`--omni_yield 200`) / 1072 (`--omni_fric 30 --omni_elas corotated`) |
| VR-GS | `python exe/baseline_vrgs.py --mode fit ...` (E by the paper's procedure), then `python exe/baseline_vrgs.py --mode run --data $D/$s.pt --tag $s --frames T <scene flags> --E E --set NxIT --out $O/vrgs_$s` | lego 167×1 (E 2.646e5), pillow2sofa 200×1 (E 7.736e3), bread 100×11 (E 4.876e3), wolf 400×11 (E 7.751e5) |
| PhysTwin | `python exe/baseline_spring.py phystwin --mode fit --phystwin $FUSEDGS_WORK/PhysTwin --data $D/$s.pt --tag $s --frames T <scene flags> --out $O/pt_${s}_prep`, then `--mode run --fit $O/pt_${s}_prep --N N --out $O/pt_$s` | N = 162 / 143 / 200 / 400 |
| Spring-Gaus | `python exe/evaluation.py multiview --sim $D/gt_$s --config CFG --shape $s --out $O/sg_${s}_prep/mv`, then `python exe/baseline_spring.py springgaus --mode prep --sg $FUSEDGS_WORK/Spring-Gaus --fit $O/sg_${s}_prep --mv $O/sg_${s}_prep/mv --data $D/$s.pt --tag $s --frames T <scene flags> --out $O/sg_${s}_prep`, `--mode train` (same arguments), `--mode run --N 200 --out $O/sg_$s` | N = 200 |

The same `evaluation.py phys / roll-to-h5 / render / pareto-vis` commands give the baselines' metrics and renders (`--method <name>`, `--pred <run>/pred.pt`).

### 4.2 Design ablation and robustness (Tab. ablation, Tab./Fig. robustness)

```bash
for s in lego pillow2sofa bread wolf; do
  for st in gauss1 gauss1_timing no_ftransfer gauss27 gauss64 cell0.75 cell1.5 cell2 \
            substeps1 substeps2 substeps4 substeps8 substeps16 substeps32 substeps64 substeps128; do
    python exe/run_fusedgs.py --scene $s --setting $st --data $D/$s.pt --gt $D/gt_$s --out $O/${st}_$s
  done
done
```

The reference row is `default` (Section 4.1). Error and time come from the runs. The 1³ time column comes from `gauss1_timing`, and min det F and ΔV come from `evaluation.py phys`. The zero-energy modes come from the first-frame stiffness matrix that each run saves:

```bash
python exe/evaluation.py zero-energy-modes --pred $O/gauss1_bread/pred.pt --solver $O/gauss1_bread/q27solver.pt --q 1 --frames 0 --out $O/gauss1_bread/eig.json
```

Use `--q 1 / 8 / 27 / 64` for 1³ / 2³ (default) / 3³ / 4³ Gauss points.

### 4.3 Substep stability (Tab. stability)

Scenes: lego, pillow2sofa, ship_plastic_lego, and mic_metal_lego in `$FUSEDGS_WORK/stab/<scene>` (ship and mic also need `gt/`, from `tools.py pg-h5-to-pt`). The driver raises the number of substeps N in powers of two, scans the integers below the first stable power, and confirms the smallest stable N three times. It does this for PhysGaussian, i-PhysGaussian, OmniPhysGS, and FusedGS (`exe/run_fusedgs.py --setting stability --N n`):

```bash
python exe/tools.py hex-input $FUSEDGS_WORK/stab/lego $FUSEDGS_WORK/stab/lego/hex_in.pt 60
python exe/tools.py stability-ladder --scene lego --stages main,vis,speed
python exe/evaluation.py analyze --scene lego --out $FUSEDGS_WORK/stab/lego/an.json
python exe/evaluation.py extra --scene lego
```

One FusedGS run of the ladder, as an example:

```bash
python exe/run_fusedgs.py --scene lego --setting stability --N 4 --data $FUSEDGS_WORK/stab/lego/hex_in.pt \
    --gt $FUSEDGS_WORK/stab/lego/rr --gt_cfg $FUSEDGS_WORK/stab/lego/config.json --out $O/stab_lego_N4
```

A run is stable if it finishes all frames, the total energy satisfies (E−E0)/|E0| ≤ 1, and floor penetration is at most 0.5 L.

### 4.4 Expressiveness (Tab. L2 tracking)

Targets come from the ground-truth trajectories of bread_foam_drop_lego, drums_honey_lego, and chair_paste_lego. The wolf target is a Gaussian velocity field (`expressiveness.py flow --pts ...`). Then each representation tracks the target:

```bash
s=bread_foam_drop_lego
mkdir -p $O/rep_$s
python exe/expressiveness.py gf-target --h5_dir <ground truth of $s>/simulation_ply --name $s --out_dir $O/rep_in_$s
for m in "hex --riem_energy logbarrier --riem_lr 0.5 --hex_dx 0.02" \
         "phystwin --riem_energy elastic --riem_lr 10 --riem_k 1 --pt_realtime" \
         "vrgsx --riem_energy logbarrier --riem_lr 1" \
         "springgaus --riem_energy elastic --riem_lr 100"; do
  set -- $m
  python exe/expressiveness.py track --opt riem --tb none --riem_eps 1e-3 --flow $O/rep_in_$s/gflow_$s.npz --aux $O/rep_in_$s/aux_$s.npz \
      --method $m --out $O/rep_$s/$1.npz
done
```

The RMSE is printed at the end of each run. `hex` is the FusedGS representation; the others are PhysTwin, VR-GS, and Spring-Gaus.

## 5. Notes

- A PhysGaussian / i-PhysGaussian run below its stable substep count stops at the first substep where particles leave the grid (`[diverged] ...`), saves the frames done so far and exits normally; `evaluation.py` and the stability driver count it as a blow-up.
- wolf with 2 or more substeps (sand) runs with two safeguards that are inactive in normal frames: particles that leave the ground-truth MPM domain are put back at frame end, and a frame in which a particle moves much faster than its neighbourhood is solved again (up to 3 times). They prevent the occasional blow-up of the 128-substep run; the log ends with how often they acted.
- Results are not bit-identical from run to run, because of the per-frame time budget and floating-point atomics on the GPU. Repeated runs of the original code differ by about the same amount as the minimized code differs from it.
- The settings table `cfg/settings.json` was generated from the command generators of the original experiment code. Diagnostic-only components of those runs (per-frame logging) are switched off; we checked that this does not change the results.
