# v3 front-end — single image → multi-view video → SC-GS 4D reconstruction

## What v3 changes vs v1/v2

v1/v2 generated 4D by **per-scene video-SDS** distilling motion into a GNN
(DreamPhysics fork, SVD prior). v3 takes the **reconstruction** route instead:
synthesize a set of **N synchronized novel-view video streams with KNOWN camera
poses** from one input image, then hand them to a **multi-view dynamic 3DGS
reconstructor (SC-GS)**. No SDS in this front-end — the motion + views come from
a pretrained multi-view video diffusion model.

```
subject.png ─▶ [SV4D 2.0] ─▶ V views × T frames (known azimuths)
                              + per-view poses + per-frame times ─▶ [SC-GS] ─▶ 4D GS
```

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

Geometry: camera at `(r·cosE·sinA, r·sinE, r·cosE·cosA)` looking at origin, up
`+y`; `--radius 2.0`, `--fov_deg 33.8` (SV4D's orbital renders are ~30–40°;
tunable — only relative consistency matters, absolute scale is a gauge the
reconstructor absorbs). Times are **identical across views** (streams are
synchronized), so timestamp `t` means the same instant in every view — exactly
what a multi-view 4D reconstructor assumes.

## Interface contract with the SC-GS stage

SC-GS consumes **posed multi-view image sequences with timestamps**. Its dynamic
pipeline is built on the D-NeRF loader, so the `transforms_{train,test}.json`
(c2w `transform_matrix` + per-frame `time`) is the primary contract; `cameras.json`
gives explicit `K`/`[R|t]` for any COLMAP-style path. SC-GS then fits canonical
3DGS + sparse control nodes + RBF-LBS warp against these posed frames — the same
representation v1 planned to drive with a GNN, now **initialized from real
multi-view supervision** instead of SDS.

- Coordinate frame: `transform_matrix` is blender/OpenGL c2w (SC-GS/D-NeRF native).
- Time: normalized `[0,1]`, shared across views.
- Views: 5 total for `sv4d2` (input + 4), 9 for `sv4d2_8views` (input + 8); view 0
  is the input azimuth.

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
  with `num_steps` (50 default; 20 is faster/lower quality) × frames × views —
  order minutes to tens of minutes per subject on an A100.

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
- **Fixed, sparse view set.** Only 4 (or 8) novel azimuths at a single elevation
  (0°). No elevation diversity → top/bottom of the object are unconstrained for
  the reconstructor.
- **Absolute camera scale is a gauge.** We emit a consistent-but-assumed FOV/radius;
  verify against the SC-GS scene scale (`--fov_deg`, `--radius`) if reconstruction
  geometry looks squashed/stretched.
- **VRAM.** 576² × 21 frames is heavy; drop to `sv4d2_8views` (T=5) or
  `--img_size=512` / `--encoding_t=1 --decoding_t=1` on ≤24 GB.
```
