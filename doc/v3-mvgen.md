# v3 front-end — single image → multi-view video → SC-GS 4D reconstruction

## What v3 changes vs v1/v2

v1/v2 generated 4D by **per-scene video-SDS** distilling motion into a GNN
(DreamPhysics fork, SVD prior). v3 takes the **reconstruction** route instead:
synthesize a set of **N synchronized novel-view video streams with KNOWN camera
poses** from one input image, then hand them to a **multi-view dynamic 3DGS
reconstructor (SC-GS)**. No SDS in this front-end — the motion + views come from
a pretrained multi-view video diffusion model.

```
subject.png ─▶ [SV4D 2.0 × N elevations] ─▶ (n_az × n_elev) views × T frames
                              + per-view poses + per-frame times ─▶ [SC-GS] ─▶ 4D GS
```

**Default is now a DENSE grid: 8 azimuths × 3 elevations = 24 views** (variant
`sv4d2_8views`, `--elevations -20,0,20`). Earlier the front-end ran SV4D **once**
at a single elevation (0°) → only 4 spatial views. That was too sparse: SC-GS's
dense-bootstrap gaussians collapsed to 0 points and the run **crashed**. Running
SV4D once per elevation and merging with elevation-aware poses fixes this.

## Model survey — what is ACTUALLY runnable today (2026-07)

| Model | Multi-view *video*? | Open weights? | Usable now |
|---|---|---|---|
| **SV4D 2.0** (Stability) | **yes** (V views × T frames) | **yes** — `stabilityai/sv4d2.0` | **RECOMMENDED** |
| SV4D (Stability, 07/2024) | yes (5×8) | yes — `stabilityai/sv4d` (+ SV3D) | usable, superseded by 2.0 |
| **CAT4D** (Google DeepMind, CVPR'25) | yes | **NO** | **not usable** — code/weights unreleased |
| SV3D (Stability) | static orbit only (no time) | yes | no (single timestep) |
| DimensionX | multi-view *or* video, LoRA-controlled | partial | fragile for synced MV-video |
| 4Diffusion | 4D from mono video | research code, weak weights | not production-ready |

**CAT4D status (verified):** the only public artifact is the project page repo
`github.com/cat-4d/cat-4d.github.io` (a GitHub Pages site). No inference code, no
checkpoints. Treat CAT4D as **unavailable**.

**Recommendation: SV4D 2.0.** It is the only open-weight model that emits
*synchronized multi-view video with a known, fixed camera set*, self-contained
(no SV3D bootstrap needed, unlike SV4D v1), with a working sampler in
`Stability-AI/generative-models`.

## SV4D 2.0 — exact input/output spec

Repo: `Stability-AI/generative-models` ·
script: `scripts/sampling/simple_video_sample_4d2.py` (Fire CLI) ·
weights: `stabilityai/sv4d2.0`.

**Input:** a **single-view (monocular) video** of one object — mp4/gif, a folder
of frames, or a filename pattern. (SV4D 2.0 does *not* take a bare still; the
front-end first animates the still with SVD-XT img2vid, then feeds those frames.)
Object should be centered on a plain background (`--remove_bg=True` uses rembg).

**Output:** per-view mp4s written to
`{output_folder}/{model_name}/{base_count:06d}_v{view:03d}.mp4`
(one video per view — **not** a grid).

**Only the NOVEL views are saved** (verified in `simple_video_sample_4d2.py`:
the save loop iterates `view_indices = arange(V)+1`). The input view (view 0,
azimuth 0) is **not** emitted as an mp4. So the sorted files `_v001.._v00V` map to
`azimuths_deg[1:]` (i.e. `[30,75,…,330]` for 8views), **not** `azimuths_deg[0:V]`.
`gen_mv_video.py`'s `novel_azimuths()` uses `[1:]`; the previous packer used
`[0:V]`, an **off-by-one** that mis-labeled every view's azimuth — now fixed.

**`elevations_deg` is a single elevation per run**, broadcast to every view
(`elevations_deg = [E]*n_views` in the sampler), so all V saved novel views are
rendered at absolute elevation `E`. To get elevation diversity we therefore call
the sampler **once per elevation** (`--elevations_deg=E`) and merge the runs. Each
elevation writes to a **distinct output subfolder** (`_work/sv4d2_out/elev_pXX` /
`elev_mXX`) so the identically-named `{model_name}/{base_count}_v{view}.mp4` files
from different elevations never overwrite one another.

**Two checkpoints (variant chosen by `--model_path` basename):**

| variant | ckpt | native T | novel V | azimuths_deg (view 0 = input) | elev | res |
|---|---|---|---|---|---|---|
| `sv4d2` | `sv4d2.safetensors` | 12 | 4 | `[0, 60, 120, 180, 240]` | 0.0 | 576² |
| `sv4d2_8views` | `sv4d2_8views.safetensors` | 5 | 8 | `[0, 30, 75, 120, 165, 210, 255, 300, 330]` | 0.0 | 576² |

`simple_video_sample_4d2.py` defaults: `num_steps=50`, `img_size=576`,
`n_frames=21` (native T extended auto-regressively so all views run the full
input length), `elevations_deg=0.0`, `seed=23`, `encoding_t=8`, `decoding_t=4`,
`azimuths_deg=None` → falls back to the fixed table above.

**sample() signature (verbatim):**
```python
def sample(input_path="assets/sv4d_videos/camel.gif",
           model_path="checkpoints/sv4d2.safetensors", output_folder="outputs",
           num_steps=50, img_size=576, n_frames=21, seed=23,
           encoding_t=8, decoding_t=4, device="cuda",
           elevations_deg=0.0, azimuths_deg=None,
           image_frame_ratio=0.9, verbose=False, remove_bg=False): ...
```

(For reference, **SV4D v1**: `simple_video_sample_4d.py`, output 5 frames × 8
views @576², 8 azimuths subsampled from a 21-way orbit at indices
`[2,5,7,9,12,14,16,19]`, `elevations_deg=10.0`, needs an SV3D pass —
`sv3d_u`/`sv3d_p` — on frame 0. Superseded; we default to 2.0.)

## Camera-pose convention we emit

SV4D orbits a **normalized, origin-centered object at fixed elevation** while
azimuth steps through the known per-view angles. SV4D does **not** emit metric
intrinsics or radius — those are a rendering convention. What a 4D reconstructor
needs is that **all views share one consistent pinhole model**, which we
guarantee. `exe/gen_mv_video.py` writes:

- **`transforms_train.json` / `transforms_test.json`** — D-NeRF/blender format
  (`camera_angle_x` + `frames[]` each with `transform_matrix` = OpenGL c2w,
  `file_path`, and `time ∈ [0,1]`). This is what SC-GS's dynamic (D-NeRF-style)
  loader ingests directly.
- **`cameras.json`** — per-view explicit `K` (3×3), `c2w_opengl`, `c2w_opencv`,
  `w2c_opencv` (`[R|t]`, cam looks +z), azimuth/elevation, image size.
- **`timestamps.json`** — `T` shared per-frame times in `[0,1]`, `num_views`, fps.
- **`images/view_KK/TTTTT.png`** — decoded frames, per view.

**Elevation-aware camera math (the key fix).** Each view now carries its OWN
`(azimuth A, elevation E)` and the c2w is built from BOTH (previously the whole
ring shared one elevation). For a view at `(A, E)`:

```
eye = ( r·cos E · sin A ,  r·sin E ,  r·cos E · cos A )        # position on the sphere
z   = normalize(eye - origin)          # camera +z (points away from target, OpenGL)
x   = normalize(world_up × z),  world_up = (0,1,0)
y   = z × x
c2w = [ x | y | z | eye ]  (4×4, OpenGL/blender transform_matrix)
```

Elevation `E` enters through `eye`: `+E` lifts the camera (`+y`) and shrinks the
horizontal radius by `cos E`; the look-at then tilts the camera **down** toward
the origin (and `−E` tilts it up). At `E=0` this reduces to the old azimuth-only
ring. Sanity-checked numerically: `|eye| = r` for all `(A,E)`, camera forward
`−z` dotted with `−êye = 1.0` (looks exactly at origin), `det(R)=1`.

`--radius 2.0`, `--fov_deg 33.8` (SV4D's orbital renders are ~30–40°; tunable —
only relative consistency matters, absolute scale is a gauge the reconstructor
absorbs). Times are **identical across views** (streams are synchronized), so
timestamp `t` means the same instant in every view — exactly what a multi-view 4D
reconstructor assumes. The shared per-frame `time ∈ [0,1]` (T timesteps) is the
same across all views and all elevations.

**View indexing.** Views are enumerated `view_00 .. view_{V−1}` across the full
`azimuth × elevation` grid (elevation-major: all azimuths of the first elevation,
then the next elevation, …). For the default 8×3 that is `V=24` views. The input
viewpoint (az 0, elev 0) is never in the set (SV4D doesn't save it); az 0 is not a
novel azimuth, so **no viewpoint is duplicated across elevation runs** — every
`(az, elev)` pair is a distinct camera.

## Interface contract with the SC-GS stage

SC-GS consumes **posed multi-view image sequences with timestamps**. Its dynamic
pipeline is built on the D-NeRF loader, so the `transforms_{train,test}.json`
(c2w `transform_matrix` + per-frame `time`) is the primary contract; `cameras.json`
gives explicit `K`/`[R|t]` for any COLMAP-style path. SC-GS then fits canonical
3DGS + sparse control nodes + RBF-LBS warp against these posed frames — the same
representation v1 planned to drive with a GNN, now **initialized from real
multi-view supervision** instead of SDS.

- Coordinate frame: `transform_matrix` is blender/OpenGL c2w (SC-GS/D-NeRF native).
- Time: normalized `[0,1]`, shared across views AND elevations.
- Views (default): `sv4d2_8views` × `--elevations -20,0,20` → **24 views**
  (8 novel azimuths × 3 elevations). `sv4d2` gives 4 novel az → 12 views at the
  same 3 elevations. Only novel views are emitted (the input az-0 view is not
  saved by the sampler), so no viewpoint is duplicated across elevation runs.

## Weights & how they download

```bash
huggingface-cli download stabilityai/sv4d2.0 sv4d2.safetensors        --local-dir checkpoints
huggingface-cli download stabilityai/sv4d2.0 sv4d2_8views.safetensors --local-dir checkpoints
# monocular bootstrap (SVD-XT img2vid) is pulled by diffusers on first run:
#   stabilityai/stable-video-diffusion-img2vid-xt
```
Per infra rules, set `HF_HOME=/data/huggingface` and `chmod -R 777` the cache.

## VRAM / runtime

- SVD-XT img2vid step: ~24 GB with `enable_model_cpu_offload()` (as `exe/gen_video.py`).
- SV4D 2.0 sampler: memory dominated by VAE encode/decode batching. Low-VRAM path
  is `--encoding_t=1 --decoding_t=1` (defaults 8/4) and/or `--img_size=512`;
  practically it targets a 40 GB-class GPU (A100/L40S) at 576² · 21 frames,
  fitting ≤24 GB only with the reduced `encoding_t/decoding_t`. Runtime scales
  with `num_steps` (50 default; 20 is faster/lower quality) × frames × views, and
  now **× n_elevations** (the sampler is invoked once per elevation — default 3×).
  Peak VRAM is unchanged (elevations run sequentially, not batched). Order tens of
  minutes per subject on an A100 for the default 3-elevation grid.

## `Dockerfile.mvgen` deps sketch

```dockerfile
# base: CUDA runtime + torch (project image ghcr.io/wgsong0110/anchorflow)
FROM ghcr.io/wgsong0110/anchorflow:latest
RUN git clone https://github.com/Stability-AI/generative-models /opt/generative-models
WORKDIR /opt/generative-models
# NOTE: build/install on an x86 GPU box or CI — NEVER on the arm64 host.
RUN pip3 install -r requirements/pt2.txt \
 && pip3 install . \
 && pip3 install -e "git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata"
RUN pip3 install diffusers transformers accelerate rembg imageio imageio-ffmpeg
ENV HF_HOME=/data/huggingface
# checkpoints/ (sv4d2*.safetensors) mounted or downloaded at runtime via huggingface-cli
```

## Risks / limits

- **View consistency.** SV4D is a diffusion prior; the V views are *not*
  perfectly geometry-consistent (mild floaters/identity drift across azimuths).
  SC-GS must be robust to this — expect it to act as a strong multi-view
  regularizer, not photometric ground truth.
- **Motion is inherited from the input.** SV4D animates the *given* monocular
  motion; it does not invent new self-actuation. Since our still→video bootstrap
  is SVD img2vid, the resulting 4D motion is only as good as SVD's guess — the
  weak point vs v1's action-controllable ambition. Supplying a real driving video
  (`--video`) bypasses this.
- **View density (was: fixed & sparse).** The old single-elevation run gave only
  4 novel azimuths at 0° → SC-GS's dense bootstrap collapsed to 0 gaussians and
  **crashed**. Fixed by the multi-elevation grid (default 8 az × 3 elev = 24
  views), which also constrains top/bottom of the object. Elevation coverage is
  still bounded by what SV4D renders plausibly (roughly ±20–30° off-plane);
  extreme top/bottom remain weakly constrained. Add more `--elevations` if needed.
- **Absolute camera scale is a gauge.** We emit a consistent-but-assumed FOV/radius;
  verify against the SC-GS scene scale (`--fov_deg`, `--radius`) if reconstruction
  geometry looks squashed/stretched.
- **VRAM.** 576² × 21 frames is heavy; drop to `sv4d2_8views` (T=5) or
  `--img_size=512` / `--encoding_t=1 --decoding_t=1` on ≤24 GB.
```
