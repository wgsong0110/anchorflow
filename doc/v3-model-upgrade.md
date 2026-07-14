# v3 model upgrade — img2vid (SVD-XT → Wan 2.2) and the SV4D role

Target: `exe/gen_mv_video.py`. Two-stage front end:

```
subject.png ─▶ (1) IMG2VID  ─▶ monocular video (T frames, 576²)
                              │  CURRENT: StableVideoDiffusionPipeline (SVD-XT)
                              │  UPGRADE: Wan 2.2 I2V  (diffusers WanImageToVideoPipeline)
                              ▼
              (2) MULTI-VIEW ─▶ V views × T frames (known azimuths)  ─▶ SC-GS dataset
                              │  CURRENT + KEPT: SV4D 2.0 (sgm sampler)
```

Findings below are verified against the live HF model cards + diffusers docs
(2026-07). Do NOT build/run on the arm64 host — this is a plan only.

---

## Task 1 — IMG2VID: replace SVD-XT with **Wan 2.2 I2V**

### Verdict

**Top pick: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`** (Alibaba, Apache-2.0).
Across 2026 open-source I2V comparisons Wan 2.2 is the quality/photorealism leader
and the most robust across varied inputs, with a first-class diffusers pipeline
(`WanImageToVideoPipeline`) that is confirmed to exist. It is a two-expert MoE:
`transformer` (high-noise stage) + `transformer_2` (low-noise stage), ~27B total /
14B active per step.

**Backup (practical / lighter): `Wan-AI/Wan2.2-TI2V-5B-Diffusers`** (Apache-2.0).
Dense 5B, single `WanPipeline` (pass `image=` for I2V), 720P @ 24fps, fits a 24GB
card with offload. Use this if A14B offloading is too slow or the 48GB budget is
tight — it is the safest fit-on-48GB choice.

Other candidates evaluated and why not chosen:
- **HunyuanVideo-I2V** (`hunyuanvideo-community/HunyuanVideo-I2V`, real diffusers
  `HunyuanVideoImageToVideoPipeline`): strong on physical motion; 720p original
  needs ~60GB, 480p distilled ~24GB. Viable second alternative but Wan 2.2 wins on
  input robustness for stylized/object-centric subjects. Newer `HunyuanVideo-1.5`
  exists (`hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v`) — lighter,
  worth a look, but less battle-tested in diffusers than Wan 2.2.
- **LTX-Video / LTX-2** (Lightricks): fastest, fits 16GB, but lower fidelity on
  detailed subjects — good for speed, not our quality target.
- **CogVideoX-1.5-5B-I2V, Mochi-1, Open-Sora 2.0**: all real but superseded by
  Wan 2.2 / Hunyuan on 2026 open-I2V quality rankings; no reason to pick them here.

### Exact pipeline (drop-in for `image_to_mono_frames`)

`WanImageToVideoPipeline` is verified in the diffusers Wan docs. For A14B, the
image path needs a `CLIPVisionModel` image encoder + `AutoencoderKLWan` VAE; the
two experts load automatically from the repo (`transformer` + `transformer_2`, with
the pipeline's `boundary_ratio` switching between them). Requires **diffusers from
git main** (`pip install git+https://github.com/huggingface/diffusers`).

```python
def image_to_mono_frames(image_path, frames_dir, n_frames, size, seed,
                         prompt="the object stays centered and animates with a "
                                "subtle natural motion, static camera, plain "
                                "background",
                         neg="camera pan, camera zoom, background clutter, "
                             "blurry, low quality, distorted"):
    import torch, os, numpy as np
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel

    os.makedirs(frames_dir, exist_ok=True)
    model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, vae=vae, image_encoder=image_encoder,
        torch_dtype=torch.bfloat16)

    # --- 48GB fit: group-offload both experts + text encoder (NOT plain .to) ---
    # Native single-GPU A14B is ~80GB; group offloading brings the resident set
    # to well under 48GB (diffusers gets Wan-14B to ~13GB this way). See NOTE.
    pipe.enable_model_cpu_offload()          # simplest; fits 48GB comfortably.
    # For tighter budgets use apply_group_offloading(...) / enable_group_offload(
    #   offload_type="leaf_level", use_stream=True) on transformer + transformer_2.

    # Wan wants square dims divisible by 16; 576 is (36*16). Object crop is square.
    image = Image.open(image_path).convert("RGB").resize((size, size))
    gen = torch.Generator("cuda").manual_seed(seed)

    # Wan native I2V length is 81 frames @16fps. Generate 81 then subsample to
    # n_frames (~21) for SV4D — SV4D re-times internally, only frame COUNT matters.
    out = pipe(image=image, prompt=prompt, negative_prompt=neg,
               height=size, width=size, num_frames=81,
               guidance_scale=3.5, guidance_scale_2=3.5,
               num_inference_steps=40, generator=gen).frames[0]

    idx = np.linspace(0, len(out) - 1, n_frames).round().astype(int)
    for i, j in enumerate(idx):
        out[j].resize((size, size), Image.LANCZOS).save(
            os.path.join(frames_dir, f"{i:05d}.png"))
    print(f"[mono] Wan2.2-I2V wrote {n_frames} frames -> {frames_dir}")
    return n_frames
```

Key params (verified against diffusers Wan docs / A14B card):
- pipeline: `WanImageToVideoPipeline` · model `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- dtype: transformer/pipeline `bfloat16`, VAE + image_encoder `float32`
- resolution: square `576×576` (must be /16); or 480×832 / 720×1280 native tiers
- frames: native **81** @ 16fps (subsample to ~21 for SV4D)
- steps 40, `guidance_scale=3.5` (+`guidance_scale_2` for the low-noise expert)
- license Apache-2.0

Backup (5B) call differs: use `WanPipeline.from_pretrained(model_id, vae=vae,
torch_dtype=bf16)` with `model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers"` and pass
`image=...` into `pipe(...)`; 720P, `num_frames=121`, steps 50, `guidance_scale=5.0`.

### VRAM (48GB budget)
- A14B native single-GPU ≈ 80GB → **must offload**. `enable_model_cpu_offload()`
  keeps only the active expert resident and fits 48GB with margin; group-offload
  (`offload_type="leaf_level", use_stream=True`) pushes it far lower if needed.
- 5B backup fits 24GB with offload → trivially fits 48GB, faster wall-clock.

---

## Task 2 — MULTI-VIEW / 4D (the SV4D role): **keep SV4D 2.0**

### Verdict: SV4D 2.0 remains the best OPEN option — no drop-in upgrade exists.

`stabilityai/sv4d2.0` is still, as of 2026-07, the only open-**weight** model that
emits *synchronized multi-view video with a known, fixed camera set* from a
monocular video, self-contained (no SV3D bootstrap), with a working sampler in
`Stability-AI/generative-models` (`scripts/sampling/simple_video_sample_4d2.py`).
Everything newer is either closed, adjacent (does a different job), or paper-only:

| Candidate | Role | Open weights? | Verdict |
|---|---|---|---|
| **SV4D 2.0** (`stabilityai/sv4d2.0`) | mono video → V×T MV video, known azimuths | **yes** | **KEEP — recommended** |
| **SP4D** (`stabilityai/sp4d`, 09/2025) | *builds on SV4D 2.0*; adds kinematic part seg | yes (revenue-capped license) | **not an upgrade** — RGB branch ≈ SV4D 2.0, same Objaverse distribution → same OOD/stylized weakness; value is rigging, not NVS quality |
| **CAT4D** (Google, CVPR'25) | MV video diffusion | **NO** (project page only) | unavailable (re-confirmed) |
| **L4GM** (`nv-tlabs/L4GM-official`) | mono video → 4D Gaussians (feed-forward) | yes | **wrong stage** — it's a *reconstructor* (SC-GS's role), not an MV-video generator; also Objaverse-trained → OOD-limited |
| **DimensionX** | LoRA-controlled MV *or* video | partial | fragile for synchronized MV-video |
| **GenXD / Diffusion4D** | 3D/4D scene gen | limited/unverified weights | not a clean synced-MV-video drop-in |
| **SyncMV4D, SS4D, MV-Performer, TriDiff-4D, Turbo4DGen, 4Real-Video, Vidu4D** | various 4D | paper-only / no verified public weights | not usable now |

### Honest note on the OOD/stylized weakness
The reported SV4D-2.0 degradation on stylized/OOD objects is **intrinsic**: it is
trained on Objaverse renders, and *every* open 4D model with weights (SP4D, L4GM,
Diffusion4D) shares that training distribution and the same failure mode. There is
no open model that fixes NVS-on-stylized today. Two realistic mitigations, in
priority order, **without** changing the SV4D stage:
1. **Better stage-1 input (this upgrade already helps):** Wan 2.2 produces a far
   cleaner, more coherent monocular video than SVD-XT; SV4D's MV output is only as
   good as its driving video, so upgrading stage 1 is the highest-leverage fix.
2. **`--remove_bg=True` + tight object centering** so SV4D sees an in-distribution
   object-on-plain-bg (closer to its training renders).
3. (Research, out of scope) camera-controlled video models (ReCamMaster-class) to
   synthesize orbit views — a redesign, not a drop-in; note only, not recommended.

**Decision: keep SV4D 2.0 exactly as wired in `run_sv4d2(...)`.** The multi-view
half of `gen_mv_video.py` is unchanged.

---

## Task 3 — Dependency / Docker strategy (the important part)

### The conflict (flagged explicitly)
- **SV4D 2.0 stage** (`Dockerfile.mvgen`): pinned to `torch 2.1.0 / cu118`, sgm +
  `requirements/pt2.txt` (kornia 0.6.9, open-clip, a torch-2.1-built xformers,
  transformers held low for sgm's open_clip conditioning). This is a fragile,
  torch-2.1-coupled env.
- **Wan 2.2 stage**: needs **diffusers from git main** + a **recent transformers**
  (CLIPVisionModel/UMT5 paths, `WanImageToVideoPipeline`, group offloading) and is
  happiest on **torch ≥ 2.4**. Modern diffusers + transformers pushed into the sgm
  image would very likely break sgm's open_clip/kornia/xformers pins.

Forcing both into one image means fighting the torch-2.1 xformers pin against a
torch-2.4 diffusers stack. **Do not co-install.** They never run in the same
process anyway.

### Cleanest resolution: **two images, hand off frames via disk/R2**

`gen_mv_video.py` already has the seam for this — `--video <frames-folder>` skips
stage 1 and feeds an existing monocular frame folder straight into SV4D. So:

1. **New `docker/Dockerfile.i2v`** (Wan stage), built on x86 CI only:
   ```dockerfile
   FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
   RUN apt-get update && apt-get install -y --no-install-recommends \
         git ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
   # diffusers MUST be from source for Wan 2.2 (WanImageToVideoPipeline / TI2V-5B)
   RUN pip install --no-cache-dir \
         "git+https://github.com/huggingface/diffusers" \
         transformers accelerate imageio imageio-ffmpeg ftfy \
         "numpy>=2" safetensors huggingface_hub
   ENV HF_HOME=/data/huggingface
   WORKDIR /workspace
   ```
2. **Keep `docker/Dockerfile.mvgen` unchanged** (torch-2.1 sgm / SV4D 2.0). Remove
   only the SVD-XT diffusers install if you want a leaner image — SV4D no longer
   needs a diffusers img2vid bootstrap once stage 1 is external.

3. **Split the entry point** — factor `image_to_mono_frames` out of
   `gen_mv_video.py` into a standalone `exe/gen_i2v.py` (imports diffusers/Wan
   only). Two-command run:
   ```bash
   # Image A (Dockerfile.i2v):  subject.png -> mono frames
   python exe/gen_i2v.py --image subject.png --out $WORK/mono_frames \
          --n_frames 21 --size 576 --seed 23
   # (optionally rclone copy $WORK/mono_frames -> r2:storage/result/anchorflow/... )

   # Image B (Dockerfile.mvgen): mono frames -> SV4D 2.0 -> SC-GS dataset
   python exe/gen_mv_video.py --video $WORK/mono_frames --out $DATASET \
          --variant sv4d2 --img_size 576
   ```
   `gen_mv_video.py`'s `--image` path (in-process SVD-XT) can stay as a legacy
   fallback, but the **recommended flow is the two-image split** above. If you
   keep a single `--image` convenience, guard the Wan import so it only loads in
   Image A.

4. **Frame handoff:** shared `/workspace` volume on one instance, or R2
   (`rclone copy` the `mono_frames/` folder) if the two stages run on different
   boxes — per infra rules, long transfers go through tmux + `rclone`, results to
   `r2:storage/result/anchorflow/`.

### Why this is the clean answer
- Zero dependency negotiation between torch-2.4/diffusers-main and torch-2.1/sgm.
- Each image stays minimal and independently buildable on CI (x86).
- The disk/R2 seam is already the contract SV4D consumes (`--video` frames folder),
  so no change to the multi-view + dataset-packing code (stages 2–3 of
  `gen_mv_video.py`) at all.

---

## Summary of changes
- **img2vid:** `Wan-AI/Wan2.2-I2V-A14B-Diffusers` via `WanImageToVideoPipeline`
  (bf16, 576², 81→21 frames, steps 40, gs 3.5, `enable_model_cpu_offload()` to fit
  48GB). Backup: `Wan2.2-TI2V-5B-Diffusers` via `WanPipeline(image=...)`.
- **multi-view:** **keep SV4D 2.0** (`stabilityai/sv4d2.0`) — still the best/only
  open-weight synchronized-MV-video model; OOD weakness is distributional and
  best mitigated by the stronger Wan driving video, not by any available swap.
- **infra:** new `docker/Dockerfile.i2v` (torch 2.4 / diffusers-main) for stage 1,
  existing `docker/Dockerfile.mvgen` (torch 2.1 / sgm) for stage 2, frames handed
  off via disk/R2 using the existing `--video` seam. Two images to avoid the
  diffusers-main ↔ sgm torch-2.1 dependency conflict.
</content>
</invoke>
