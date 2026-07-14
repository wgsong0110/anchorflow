#!/usr/bin/env python
"""v3 front-end: single image -> multi-view video (SV4D 2.0) -> SC-GS-ready dataset.

Pipeline (all GPU steps; NOT runnable on the arm64 host):

    subject.png
      │  (1) SVD img2vid  — animate the still into a monocular driving video
      ▼      (reuses the exe/gen_video.py recipe; skipped if --video given)
    monocular video  (T frames, object-centered, 576x576)
      │  (2) SV4D 2.0  — Stability-AI/generative-models, self-contained
      ▼      scripts/sampling/simple_video_sample_4d2.py
    V synchronized novel-view videos  (V views x T frames, KNOWN azimuths)
      │  (3) pose + pack
      ▼
    dataset/  — per-view frame folders
              + transforms_train.json (D-NeRF/blender: c2w + per-frame time)
              + cameras.json          (per-view K + OpenCV/OpenGL extrinsics)
              + timestamps.json        (shared per-frame times)

SV4D 2.0 output geometry (verbatim from the repo configs / sampling script):

  variant "sv4d2"        : model_path=checkpoints/sv4d2.safetensors
                           T=12 native frames, V=4 novel views
                           azimuths_deg = [0, 60, 120, 180, 240]  (view 0 = input)
                           elevations_deg = 0.0, img_size = 576
  variant "sv4d2_8views" : model_path=checkpoints/sv4d2_8views.safetensors
                           T=5 native frames, V=8 novel views
                           azimuths_deg = [0, 30, 75, 120, 165, 210, 255, 300, 330]
                           elevations_deg = 0.0, img_size = 576

  simple_video_sample_4d2.py default n_frames=21 (native T extended
  auto-regressively); output written per-view as:
      {output_folder}/{model_name}/{base_count:06d}_v{view:03d}.mp4

Usage:
    python exe/gen_mv_video.py --image subject.png --out /path/to/dataset
    python exe/gen_mv_video.py --image subject.png --out ds --variant sv4d2_8views
    python exe/gen_mv_video.py --video mono.mp4    --out ds   # skip SVD step
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# SV4D 2.0 variant table — the KNOWN camera azimuths of the generated views.
# Numbers taken from Stability-AI/generative-models configs/sv4d2*.yaml and
# scripts/sampling/simple_video_sample_4d2.py. azimuths_deg[0] is the input view.
# --------------------------------------------------------------------------- #
SV4D2_VARIANTS = {
    "sv4d2": {
        "ckpt": "sv4d2.safetensors",
        "azimuths_deg": [0.0, 60.0, 120.0, 180.0, 240.0],
        "native_frames": 12,
    },
    "sv4d2_8views": {
        "ckpt": "sv4d2_8views.safetensors",
        "azimuths_deg": [0.0, 30.0, 75.0, 120.0, 165.0, 210.0, 255.0, 300.0, 330.0],
        "native_frames": 5,
    },
}

GENMODELS_DIR = os.environ.get("GENMODELS_DIR", "/opt/generative-models")
SV4D2_SCRIPT = "scripts/sampling/simple_video_sample_4d2.py"


# --------------------------------------------------------------------------- #
# (1) single image -> monocular driving video (SVD img2vid), saved as frames
# --------------------------------------------------------------------------- #
def image_to_mono_frames(image_path, frames_dir, n_frames, size, seed):
    """Animate a still image into `n_frames` object-centered 576x576 PNGs.

    Same recipe as exe/gen_video.py (SVD-XT img2vid + cpu offload to fit 24GB),
    then center-square-crop/resize so SV4D sees a centered object at `size`x`size`.
    """
    import torch
    from PIL import Image
    from diffusers import StableVideoDiffusionPipeline

    os.makedirs(frames_dir, exist_ok=True)
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()

    image = Image.open(image_path).convert("RGB").resize((1024, 576))
    gen = torch.Generator("cuda").manual_seed(seed)
    frames = pipe(image, num_frames=n_frames, decode_chunk_size=8,
                  generator=gen).frames[0]

    for i, f in enumerate(frames):
        w, h = f.size
        s = min(w, h)
        f = f.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        f = f.resize((size, size), Image.LANCZOS)
        f.save(os.path.join(frames_dir, f"{i:05d}.png"))
    print(f"[mono] SVD wrote {len(frames)} frames -> {frames_dir}")
    return len(frames)


# --------------------------------------------------------------------------- #
# (2) monocular video -> SV4D 2.0 multi-view videos (per-view mp4s)
# --------------------------------------------------------------------------- #
def run_sv4d2(input_path, out_dir, variant, n_frames, num_steps, img_size,
              elevation_deg, seed, remove_bg, low_vram):
    """Invoke the real SV4D 2.0 sampler (Fire CLI) as a subprocess.

    Writes per-view mp4s to {out_dir}/{model_name}/{base_count:06d}_v{view:03d}.mp4
    """
    cfg = SV4D2_VARIANTS[variant]
    model_path = os.path.join("checkpoints", cfg["ckpt"])
    script = os.path.join(GENMODELS_DIR, SV4D2_SCRIPT)

    cmd = [
        sys.executable, script,
        f"--input_path={input_path}",
        f"--model_path={model_path}",
        f"--output_folder={out_dir}",
        f"--n_frames={n_frames}",
        f"--num_steps={num_steps}",
        f"--img_size={img_size}",
        f"--elevations_deg={elevation_deg}",
        f"--seed={seed}",
    ]
    if remove_bg:
        cmd.append("--remove_bg=True")
    if low_vram:
        cmd += ["--encoding_t=1", "--decoding_t=1"]
    # NOTE: azimuths_deg defaults to None -> the script uses the variant's fixed
    # azimuths (SV4D2_VARIANTS[variant]["azimuths_deg"]). We rely on that default
    # so the emitted poses (below) match the generated views exactly.

    print("[sv4d2] running:", " ".join(cmd))
    print("[sv4d2] cwd:", GENMODELS_DIR)
    subprocess.run(cmd, cwd=GENMODELS_DIR, check=True)

    model_name = os.path.splitext(cfg["ckpt"])[0]              # "sv4d2"
    view_mp4s = sorted(glob.glob(os.path.join(out_dir, model_name, "*_v*.mp4")))
    if not view_mp4s:
        raise RuntimeError(
            f"[sv4d2] no per-view mp4s under {out_dir}/{model_name} — "
            "verify the sampler ran and the output naming "
            "({base_count:06d}_v{view:03d}.mp4).")
    print(f"[sv4d2] {len(view_mp4s)} per-view videos:")
    for p in view_mp4s:
        print("        ", p)
    return view_mp4s


def decode_mp4(path):
    """Return list of HxWx3 uint8 frames from an mp4 (imageio/ffmpeg backend)."""
    import imageio.v3 as iio
    return list(iio.imread(path))                              # (T,H,W,3)


# --------------------------------------------------------------------------- #
# (3) camera geometry — orbital poses for the known azimuths / elevation
# --------------------------------------------------------------------------- #
def look_at_c2w(eye, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    """OpenGL/blender camera-to-world (camera looks down -z, +y up).

    This is the `transform_matrix` convention consumed by D-NeRF / SC-GS.
    """
    eye = np.asarray(eye, np.float64)
    target = np.asarray(target, np.float64)
    up = np.asarray(up, np.float64)
    z = eye - target                                          # backward (+z)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def orbital_pose(azimuth_deg, elevation_deg, radius):
    """Camera on a sphere of `radius`, looking at the origin.

    SV4D orbits a normalized, origin-centered object at fixed elevation while
    azimuth steps through the KNOWN per-view angles. The absolute radius/FOV are
    a rendering convention (not emitted by SV4D); what a 4D reconstructor needs
    is that all views share ONE consistent camera model — which they do here.
    """
    a = np.deg2rad(azimuth_deg)
    e = np.deg2rad(elevation_deg)
    eye = np.array([radius * np.cos(e) * np.sin(a),
                    radius * np.sin(e),
                    radius * np.cos(e) * np.cos(a)])
    return look_at_c2w(eye)


def intrinsics(fov_deg, size):
    f = 0.5 * size / np.tan(0.5 * np.deg2rad(fov_deg))
    return np.array([[f, 0.0, size / 2.0],
                     [0.0, f, size / 2.0],
                     [0.0, 0.0, 1.0]])


# OpenGL(c2w) -> OpenCV(c2w): flip the y and z camera axes.
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


# --------------------------------------------------------------------------- #
# dataset writer — SC-GS / D-NeRF compatible
# --------------------------------------------------------------------------- #
def write_dataset(out, per_view_frames, azimuths_deg, elevation_deg,
                  radius, fov_deg, size, fps):
    """per_view_frames: list (len V) of lists (len T) of HxWx3 uint8 frames."""
    from PIL import Image

    V = len(per_view_frames)
    T = min(len(v) for v in per_view_frames)
    times = [i / (T - 1) if T > 1 else 0.0 for i in range(T)]
    K = intrinsics(fov_deg, size)
    fov_x = np.deg2rad(fov_deg)

    img_root = os.path.join(out, "images")
    os.makedirs(img_root, exist_ok=True)

    dnerf_frames, cams = [], []
    for k in range(V):
        az = azimuths_deg[k] if k < len(azimuths_deg) else (360.0 * k / V)
        c2w_gl = orbital_pose(az, elevation_deg, radius)
        c2w_cv = c2w_gl @ _GL2CV
        w2c_cv = np.linalg.inv(c2w_cv)

        vdir = os.path.join(img_root, f"view_{k:02d}")
        os.makedirs(vdir, exist_ok=True)
        for t in range(T):
            fp = os.path.join(vdir, f"{t:05d}.png")
            Image.fromarray(per_view_frames[k][t]).save(fp)
            dnerf_frames.append({
                "file_path": f"./images/view_{k:02d}/{t:05d}",   # D-NeRF: no ext
                "time": times[t],
                "transform_matrix": c2w_gl.tolist(),             # OpenGL c2w
            })

        cams.append({
            "view": k,
            "azimuth_deg": az,
            "elevation_deg": elevation_deg,
            "img_size": [size, size],
            "K": K.tolist(),
            "c2w_opengl": c2w_gl.tolist(),
            "c2w_opencv": c2w_cv.tolist(),
            "w2c_opencv": w2c_cv.tolist(),
        })

    # (a) D-NeRF / blender transforms — the format SC-GS's dynamic loader eats.
    with open(os.path.join(out, "transforms_train.json"), "w") as f:
        json.dump({"camera_angle_x": float(fov_x), "frames": dnerf_frames},
                  f, indent=2)
    # SC-GS also opens transforms_test.json; reuse train so the loader is happy.
    with open(os.path.join(out, "transforms_test.json"), "w") as f:
        json.dump({"camera_angle_x": float(fov_x), "frames": dnerf_frames},
                  f, indent=2)
    # (b) explicit per-view intrinsics + extrinsics (both conventions).
    with open(os.path.join(out, "cameras.json"), "w") as f:
        json.dump({"fov_deg": fov_deg, "radius": radius,
                   "convention": "c2w_opengl is blender transform_matrix; "
                                 "w2c_opencv is [R|t] world->cam, cam looks +z",
                   "cameras": cams}, f, indent=2)
    # (c) shared per-frame timestamps.
    with open(os.path.join(out, "timestamps.json"), "w") as f:
        json.dump({"num_frames": T, "num_views": V, "fps": fps,
                   "times": times}, f, indent=2)

    print("\n[dataset] layout ------------------------------------------------")
    print(f"  {out}/")
    print(f"    images/view_00 .. view_{V-1:02d}/   ({T} frames each)")
    print(f"    transforms_train.json  (D-NeRF: {V*T} frame entries, per-frame time)")
    print(f"    transforms_test.json   (copy of train)")
    print(f"    cameras.json           (per-view K + c2w/w2c)")
    print(f"    timestamps.json        (T={T} times in [0,1], shared across views)")
    print(f"  views V={V}  frames T={T}  size={size}x{size}  "
          f"elev={elevation_deg}  fov={fov_deg}deg  radius={radius}")
    print(f"  azimuths(deg)={[round(azimuths_deg[k] if k < len(azimuths_deg) else 360.0*k/V, 1) for k in range(V)]}")
    print("---------------------------------------------------------------")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="single input image -> SVD monocular video -> SV4D")
    src.add_argument("--video", help="existing monocular video (mp4/gif) or frames folder; skip SVD")
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--variant", default="sv4d2", choices=list(SV4D2_VARIANTS),
                    help="sv4d2 (V=4,T=12) or sv4d2_8views (V=8,T=5)")
    ap.add_argument("--n_frames", type=int, default=21,
                    help="SV4D2 output frames per view (native T extended auto-regressively)")
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--img_size", type=int, default=576)
    ap.add_argument("--elevation_deg", type=float, default=0.0,
                    help="SV4D2 elevation (default 0.0, matches training)")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--remove_bg", action="store_true", help="rembg on plain-bg input")
    ap.add_argument("--low_vram", action="store_true", help="encoding_t=1 decoding_t=1")
    # camera-model knobs (rendering convention SV4D does not emit; kept consistent
    # across all views so the reconstructor sees one pinhole model).
    ap.add_argument("--radius", type=float, default=2.0, help="orbit radius (world units)")
    ap.add_argument("--fov_deg", type=float, default=33.8,
                    help="assumed pinhole FOV for emitted intrinsics (tunable; "
                         "SV4D's orbital renders are ~30-40deg — verify vs your SC-GS scale)")
    ap.add_argument("--fps", type=float, default=10.0, help="timestamp fps metadata")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "_work")
    os.makedirs(work, exist_ok=True)

    # (1) obtain the monocular driving video (frames folder that SV4D accepts).
    if args.image:
        mono_dir = os.path.join(work, "mono_frames")
        image_to_mono_frames(args.image, mono_dir, args.n_frames,
                             args.img_size, args.seed)
        sv4d_input = mono_dir
    else:
        sv4d_input = args.video

    # (2) SV4D 2.0 -> per-view mp4s.
    sv4d_out = os.path.join(work, "sv4d2_out")
    view_mp4s = run_sv4d2(sv4d_input, sv4d_out, args.variant, args.n_frames,
                          args.num_steps, args.img_size, args.elevation_deg,
                          args.seed, args.remove_bg, args.low_vram)

    # (3) decode + pack into a posed, timestamped dataset.
    per_view_frames = [decode_mp4(p) for p in view_mp4s]
    write_dataset(args.out, per_view_frames,
                  SV4D2_VARIANTS[args.variant]["azimuths_deg"],
                  args.elevation_deg, args.radius, args.fov_deg,
                  args.img_size, args.fps)

    print(f"\n[gen_mv_video] done -> {args.out}")


if __name__ == "__main__":
    main()
