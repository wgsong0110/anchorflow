#!/usr/bin/env python3
"""
Compute RAFT optical flow between consecutive frames of a NeRF-Blender dataset.
Output: {vid_dir}/flows/{view}/flow_{i:04d}.npy  shape [H, W, 2] float32 pixel displacement
"""
import os, json, argparse
import numpy as np
import torch
from PIL import Image
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid_dir", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).cuda().eval()
    transforms = weights.transforms()

    meta = json.load(open(os.path.join(args.vid_dir, f"transforms_{args.split}.json")))
    by_view = defaultdict(list)
    for f in meta["frames"]:
        view = f["file_path"].split("/")[-2]
        by_view[view].append(f)

    out_base = os.path.join(args.vid_dir, "flows")
    for view, frames in sorted(by_view.items()):
        frames_sorted = sorted(frames, key=lambda x: x["time"])
        view_out = os.path.join(out_base, view)
        os.makedirs(view_out, exist_ok=True)

        imgs = []
        for fr in frames_sorted:
            path = os.path.join(args.vid_dir, fr["file_path"] + ".png")
            img = np.array(Image.open(path).convert("RGB"))
            imgs.append(torch.from_numpy(img).permute(2, 0, 1).float())  # [3, H, W]

        for i in range(len(imgs) - 1):
            out_path = os.path.join(view_out, f"flow_{i:04d}.npy")
            if os.path.exists(out_path):
                continue
            img1_t, img2_t = transforms(imgs[i].unsqueeze(0), imgs[i + 1].unsqueeze(0))
            with torch.no_grad():
                flow = model(img1_t.cuda(), img2_t.cuda())[-1][0]  # [2, H, W]
            np.save(out_path, flow.permute(1, 2, 0).cpu().numpy().astype(np.float32))

        print(f"[gen_flow] {view}: {len(imgs)-1} flows → {view_out}")

    print("[gen_flow] done")


if __name__ == "__main__":
    main()
