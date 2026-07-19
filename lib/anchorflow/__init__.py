"""anchorflow — anchor-based 4D scene generation.

Modules
    anchors         anchor node extraction and LBS binding (AnchorSet)
    seqgen          non-autoregressive Spatial-Temporal Transformer trajectory model
    graph           anchor graph construction (knn / radius) — used for ARAP
    warp            LBS warp of 3DGS Gaussians
    sds             SVD Motion Distillation Sampling (MDS) guidance
    tokens_to_nodes semantic anchor allocation via DINOv2 + dynamic tendency
    checkpoint      CheckpointManager
"""
