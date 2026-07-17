"""Score-Distillation-Sampling (SDS) guidance from Stable Video Diffusion.

SVD specifics:
  - UNetSpatioTemporalConditionModel; latents [B, T, C, H, W].
  - EulerDiscreteScheduler, v_prediction, continuous timestep = 0.25*ln(sigma).
  - Conditioning: CLIP image embed + VAE-encoded cond frame (concat on channel dim).
  - CFG: uncond half has zeroed image embed/latent.

MDS (DreamPhysics, AAAI'25):
    grad = w(sigma) * (eps_pred(video) - eps_pred(static_frame0))

Optimisations:
  - UNet + image_encoder on GPU permanently (no CPU offload overhead)
  - encode_frames: batch T frames in one VAE call (OOM fallback to per-frame)
  - static frame-0: encode once, repeat in latent space
  - MDS: batch eps_dyn+eps_stat into a single UNet call (batch=4)
    with OOM fallback to 2 sequential calls (batch=2 each)
"""

from __future__ import annotations

import gc
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SVDGuidance:
    def __init__(self, model_id="stabilityai/stable-video-diffusion-img2vid-xt",
                 device="cuda", dtype=torch.float16,
                 sigma_min=0.05, sigma_max=20.0, guidance_scale=3.0,
                 motion_bucket_id=127, fps=7, noise_aug=0.02,
                 use_vae_scaling=True, grad_clip=1.0):
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, variant="fp16")
        # VAE: GPU fp32 (grad flows through encode)
        self.vae = pipe.vae.to(device, torch.float32).eval()
        # UNet + image_encoder: GPU fp16 (frozen, no transfer overhead)
        self.unet = pipe.unet.to(device, dtype).eval()
        self.image_encoder = pipe.image_encoder.to(device, dtype).eval()
        self.feature_extractor = pipe.feature_extractor
        for m in (self.vae, self.unet, self.image_encoder):
            m.requires_grad_(False)
        self.device, self.dtype = device, dtype
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.guidance_scale = guidance_scale
        self.motion_bucket_id, self.fps, self.noise_aug = motion_bucket_id, fps, noise_aug
        self.vae_scale = pipe.vae.config.scaling_factor if use_vae_scaling else 1.0
        self.num_frames = self.unet.config.num_frames
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            print("[SVDGuidance] xformers attention enabled")
        except Exception:
            try:
                self.unet.set_attention_slice("auto")
                print("[SVDGuidance] attention slicing enabled (xformers unavailable)")
            except Exception:
                pass
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        self.grad_clip = grad_clip

    # --- encoders --------------------------------------------------------- #
    @torch.no_grad()
    def _clip_embed(self, cond_image):
        """cond_image: [3,H,W] in [0,1]. -> [1,1,1024] fp16."""
        img = cond_image[None] * 2 - 1
        img = F.interpolate(img, (224, 224), mode="bilinear", align_corners=False)
        img = (img + 1) / 2
        px = self.feature_extractor(images=img, do_normalize=True,
                                    do_center_crop=False, do_resize=False,
                                    do_rescale=False, return_tensors="pt").pixel_values
        emb = self.image_encoder(px.to(self.device, self.dtype)).image_embeds
        return emb.unsqueeze(1)                                  # [1,1,1024]

    @torch.no_grad()
    def _cond_latent(self, cond_image):
        """VAE-encode noise-augmented conditioning frame -> [1,4,h,w] float32."""
        img = (cond_image[None] * 2 - 1).to(self.device).float()
        img = img + self.noise_aug * torch.randn_like(img)
        return self.vae.encode(img).latent_dist.mode()

    def encode_frames(self, frames, use_checkpoint=True):
        """frames: [T,3,H,W] in [0,1], WITH grad. -> [1,T,4,h,w] float32.

        The VAE encoder stores ~12GB of conv activations for T=25 at 256x168 —
        by far the largest allocation in a training step. Checkpointing it costs
        one extra encoder forward in backward (~124ms) but frees that 12GB,
        which is what lets the rasteriser keep its graph instead of being
        checkpointed (~267ms of recompute per step).
        """
        x = (frames * 2 - 1).float()

        def _enc(z):
            return self.vae.encode(z).latent_dist.mode()

        try:
            if use_checkpoint and x.requires_grad:
                lat = checkpoint(_enc, x, use_reentrant=False) * self.vae_scale
            else:
                lat = _enc(x) * self.vae_scale
            return lat.unsqueeze(0)                              # [1,T,4,h,w]
        except RuntimeError:
            lats = [self.vae.encode(x[i:i+1]).latent_dist.mode() * self.vae_scale
                    for i in range(x.shape[0])]
            return torch.stack(lats, dim=1)

    def _time_ids(self):
        return torch.tensor([[self.fps - 1, self.motion_bucket_id, self.noise_aug]],
                            dtype=self.dtype, device=self.device)

    def _sample_sigma(self):
        u = torch.rand(1, device=self.device)
        log_s = (torch.log(torch.tensor(self.sigma_min, device=self.device)) * (1 - u)
                 + torch.log(torch.tensor(self.sigma_max, device=self.device)) * u)
        return log_s.exp()

    # --- UNet helper ------------------------------------------------------ #
    def _unet_forward(self, z_batch, cond_lat_batch, emb_batch, tid_batch, sigma):
        """One UNet forward (arbitrary batch size). Returns v-predictions."""
        t = 0.25 * torch.log(sigma)
        z_in = (z_batch / (sigma ** 2 + 1).sqrt()).to(self.dtype)
        c_in = cond_lat_batch.to(self.dtype)
        e_in = emb_batch.to(self.dtype)
        ti_in = tid_batch.to(self.dtype)
        v = self.unet(torch.cat([z_in, c_in], dim=2), t,
                      encoder_hidden_states=e_in, added_time_ids=ti_in,
                      return_dict=False)[0]
        return v

    @torch.no_grad()
    def _eps_pred_single(self, z, sigma, cond_lat, img_emb, time_ids):
        """CFG eps prediction for one video clip. z: [1,T,4,h,w]."""
        B2_z   = torch.cat([z, z], dim=0)
        B2_cl  = torch.cat([torch.zeros_like(cond_lat), cond_lat], dim=0)
        B2_emb = torch.cat([torch.zeros_like(img_emb), img_emb], dim=0)
        B2_tid = time_ids.expand(2, -1)
        v = self._unet_forward(B2_z, B2_cl, B2_emb, B2_tid, sigma)
        v_u, v_c = v.chunk(2)
        v_cfg = v_u + self.guidance_scale * (v_c - v_u)
        x0_pred = v_cfg * (-sigma / (sigma ** 2 + 1).sqrt()) + z / (sigma ** 2 + 1)
        return (z - x0_pred) / sigma

    @torch.no_grad()
    def _eps_pred_mds(self, z_dyn, z_stat, sigma, cond_lat, img_emb, time_ids):
        """Fused CFG for both dynamic and static clips in one UNet call (batch=4).
        Falls back to two sequential calls (batch=2) on OOM."""
        zeros_cl  = torch.zeros_like(cond_lat)
        zeros_emb = torch.zeros_like(img_emb)
        tid4 = time_ids.expand(4, -1)

        def _cfg(v_batch, z_ref):
            v_u_d, v_c_d, v_u_s, v_c_s = v_batch.chunk(4)
            def _x0(v, z): return v * (-sigma / (sigma**2 + 1).sqrt()) + z / (sigma**2 + 1)
            eps_d = (z_dyn - _x0(v_u_d + self.guidance_scale * (v_c_d - v_u_d), z_dyn)) / sigma
            eps_s = (z_stat - _x0(v_u_s + self.guidance_scale * (v_c_s - v_u_s), z_stat)) / sigma
            return eps_d, eps_s

        try:
            B4_z   = torch.cat([z_dyn, z_dyn, z_stat, z_stat], dim=0)
            B4_cl  = torch.cat([zeros_cl, cond_lat, zeros_cl, cond_lat], dim=0)
            B4_emb = torch.cat([zeros_emb, img_emb, zeros_emb, img_emb], dim=0)
            v = self._unet_forward(B4_z, B4_cl, B4_emb, tid4, sigma)
            return _cfg(v, None)
        except RuntimeError:
            # OOM: fall back to two sequential CFG calls
            eps_d = self._eps_pred_single(z_dyn,  sigma, cond_lat, img_emb, time_ids)
            eps_s = self._eps_pred_single(z_stat, sigma, cond_lat, img_emb, time_ids)
            return eps_d, eps_s

    def _apply(self, x0, grad):
        grad = torch.nan_to_num(grad)
        if self.grad_clip:
            grad = grad.clamp(-self.grad_clip, self.grad_clip)
        target = (x0 - grad).detach()
        return 0.5 * F.mse_loss(x0.float(), target.float(), reduction="sum") / x0.shape[0]

    def _cond(self, cond_image, T):
        img_emb  = self._clip_embed(cond_image)
        cond_lat = self._cond_latent(cond_image).unsqueeze(1).repeat(1, T, 1, 1, 1)
        return cond_lat, img_emb, self._time_ids()

    @torch.no_grad()
    def precompute_cond(self, cond_image, T):
        """Cache the frame-0-derived terms. frame-0 is the canonical render, so
        for a fixed camera these never change across steps: CLIP embed, the
        static latent, and the conditioning latent are all recomputed every step
        otherwise."""
        lat0 = self.vae.encode(
            (cond_image[None] * 2 - 1).float()
        ).latent_dist.mode() * self.vae_scale                  # [1,4,h,w]
        return {
            "img_emb":  self._clip_embed(cond_image),
            "lat0":     lat0,
            "cond_lat": self._cond_latent(cond_image).unsqueeze(1).repeat(1, T, 1, 1, 1),
            "time_ids": self._time_ids(),
        }

    # --- SDS -------------------------------------------------------------- #
    def sds_loss(self, frames, cond_image, w_power=0.0):
        T  = frames.shape[0]
        x0 = self.encode_frames(frames)
        cond_lat, img_emb, time_ids = self._cond(cond_image, T)
        sigma = self._sample_sigma()
        with torch.no_grad():
            noise = torch.randn_like(x0)
            eps   = self._eps_pred_single(x0 + sigma * noise, sigma,
                                          cond_lat, img_emb, time_ids)
            grad  = (sigma ** w_power) * (eps - noise)
        return self._apply(x0, grad)

    # --- Motion Distillation Sampling (DreamPhysics, AAAI'25) -------------- #
    def mds_loss(self, frames, cond_image, w_power=0.0, cond_cache=None):
        """MDS: grad = w(sigma) * (eps(video) - eps(static frame-0)).
        Single UNet forward for both clips (batch=4, OOM fallback to batch=2×2).
        cond_cache: precompute_cond() output for this camera (frame-0 is fixed)."""
        T  = frames.shape[0]
        x0 = self.encode_frames(frames)                          # [1,T,4,h,w] w/ grad
        with torch.no_grad():
            if cond_cache is not None:
                lat0     = cond_cache["lat0"]
                cond_lat = cond_cache["cond_lat"]
                img_emb  = cond_cache["img_emb"]
                time_ids = cond_cache["time_ids"]
            else:
                # static: encode frame-0 once, expand in latent space
                lat0 = self.vae.encode(
                    (frames[0:1] * 2 - 1).float()
                ).latent_dist.mode() * self.vae_scale            # [1,4,h,w]
                cond_lat, img_emb, time_ids = self._cond(cond_image, T)
            x0_stat  = lat0.unsqueeze(0).expand(1, T, -1, -1, -1)  # [1,T,4,h,w]
            sigma    = self._sample_sigma()
            noise    = torch.randn_like(x0)
            z_dyn    = x0      + sigma * noise
            z_stat   = x0_stat + sigma * noise
            eps_dyn, eps_stat = self._eps_pred_mds(
                z_dyn, z_stat, sigma, cond_lat, img_emb, time_ids)
            grad = (sigma ** w_power) * (eps_dyn - eps_stat)
        return self._apply(x0, grad)
