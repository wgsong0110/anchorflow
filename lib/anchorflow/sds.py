"""Score-Distillation-Sampling (SDS) guidance from Stable Video Diffusion.

Distils SVD's motion prior into our rendered rollout: render a clip of T frames,
push its SDS gradient back through the renderer -> LBS warp -> GNN.

SVD specifics (verified against diffusers v0.31 `StableVideoDiffusionPipeline`
and the SVD-XT configs):
  - UNet = UNetSpatioTemporalConditionModel, latents are 5-D [B, T, C, H, W].
  - EulerDiscreteScheduler, v_prediction, continuous timestep = 0.25*ln(sigma).
    c_in = 1/sqrt(sigma^2+1); x0 = v*(-sigma/sqrt(sigma^2+1)) + z/(sigma^2+1)
    with implicit sigma_data = 1; noised latent z = x0 + sigma*noise.
  - Conditioning: (a) CLIP image embed -> encoder_hidden_states [B,1,1024];
    (b) VAE-encoded (noise-augmented) cond frame, broadcast over T and concatenated
    on the CHANNEL dim (dim=2) -> UNet in_channels 8 = 4 noisy + 4 cond;
    (c) added_time_ids = [fps-1, motion_bucket_id, noise_aug_strength].
  - CFG: single batched (2B) UNet call; uncond half has zeroed image embed/latent.

SDS gradient (v-prediction -> eps):
    eps_pred = (z - x0_pred) / sigma
    grad     = w(sigma) * (eps_pred - noise)      # backpropped into x0 (latents)

Uncertain bits (flagged, tunable, verify on the instance):
  * VAE scaling_factor placement for SVD's temporal VAE (encode side).
  * sigma sampling range (config sigma_max=700 is impractical; we sample a mid band).
  * guidance scale for SDS (SVD native max=3; SDS often wants more).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class SVDGuidance:
    def __init__(self, model_id="stabilityai/stable-video-diffusion-img2vid",
                 device="cuda", dtype=torch.float16,
                 sigma_min=0.05, sigma_max=20.0, guidance_scale=3.0,
                 motion_bucket_id=127, fps=7, noise_aug=0.02,
                 use_vae_scaling=True, grad_clip=1.0):
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, variant="fp16")
        self.vae = pipe.vae.to(device).eval()
        self.unet = pipe.unet.to(device).eval()
        self.image_encoder = pipe.image_encoder.to(device).eval()
        self.feature_extractor = pipe.feature_extractor
        self.video_processor = getattr(pipe, "video_processor", None)
        for m in (self.vae, self.unet, self.image_encoder):
            m.requires_grad_(False)
        self.device, self.dtype = device, dtype
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.guidance_scale = guidance_scale
        self.motion_bucket_id, self.fps, self.noise_aug = motion_bucket_id, fps, noise_aug
        self.vae_scale = pipe.vae.config.scaling_factor if use_vae_scaling else 1.0
        self.grad_clip = grad_clip
        self.num_frames = self.unet.config.num_frames

    # --- encoders --------------------------------------------------------- #
    @torch.no_grad()
    def _clip_embed(self, cond_image):
        """cond_image: [3,H,W] in [0,1]. -> encoder_hidden_states [1,1,1024]."""
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
        """VAE-encode the noise-augmented conditioning frame -> [1,4,h,w]."""
        img = (cond_image[None] * 2 - 1).to(self.device, self.dtype)
        img = img + self.noise_aug * torch.randn_like(img)
        return self.vae.encode(img).latent_dist.mode()

    def encode_frames(self, frames):
        """frames: [T,3,H,W] in [0,1], WITH grad. -> latents [1,T,4,h,w].
        Encode frame-by-frame to avoid OOM on 25-frame sequences."""
        lats = []
        for t in range(frames.shape[0]):
            x = (frames[t:t+1] * 2 - 1).to(self.dtype)   # [1,3,H,W]
            lat = self.vae.encode(x).latent_dist.mode() * self.vae_scale
            lats.append(lat)                                # [1,4,h,w]
        return torch.stack(lats, dim=1)                    # [1,T,4,h,w]

    def _time_ids(self):
        ids = torch.tensor([[self.fps - 1, self.motion_bucket_id, self.noise_aug]],
                           dtype=self.dtype, device=self.device)
        return ids

    # --- shared UNet eval ------------------------------------------------- #
    def _sample_sigma(self):
        u = torch.rand(1, device=self.device)
        log_s = torch.log(torch.tensor(self.sigma_min, device=self.device)) * (1 - u) \
            + torch.log(torch.tensor(self.sigma_max, device=self.device)) * u
        return log_s.exp()

    @torch.no_grad()
    def _eps_pred(self, z, sigma, cond_lat, img_emb, time_ids):
        """Predict eps for noised latent z at sigma with CFG. z: [1,T,4,h,w]."""
        t = 0.25 * torch.log(sigma)
        zin = z / (sigma ** 2 + 1).sqrt()
        zin2 = torch.cat([zin, zin], dim=0)
        clat2 = torch.cat([torch.zeros_like(cond_lat), cond_lat], dim=0)
        emb2 = torch.cat([torch.zeros_like(img_emb), img_emb], dim=0)
        tid2 = torch.cat([time_ids, time_ids], dim=0)
        v = self.unet(torch.cat([zin2, clat2], dim=2), t,
                      encoder_hidden_states=emb2, added_time_ids=tid2,
                      return_dict=False)[0]
        v_u, v_c = v.chunk(2)
        v = v_u + self.guidance_scale * (v_c - v_u)
        x0_pred = v * (-sigma / (sigma ** 2 + 1).sqrt()) + z / (sigma ** 2 + 1)
        return (z - x0_pred) / sigma

    def _cond(self, cond_image, T):
        img_emb = self._clip_embed(cond_image)
        cond_lat = self._cond_latent(cond_image).unsqueeze(1).repeat(1, T, 1, 1, 1)
        return cond_lat, img_emb, self._time_ids()

    def _apply(self, x0, grad):
        grad = torch.nan_to_num(grad)
        if self.grad_clip:
            grad = grad.clamp(-self.grad_clip, self.grad_clip)
        target = (x0 - grad).detach()
        return 0.5 * F.mse_loss(x0.float(), target.float(), reduction="sum") / x0.shape[0]

    # --- SDS -------------------------------------------------------------- #
    def sds_loss(self, frames, cond_image, w_power=0.0):
        """Standard SDS: grad = w(sigma) * (eps_pred - noise)."""
        T = frames.shape[0]
        x0 = self.encode_frames(frames)                         # [1,T,4,h,w] grad
        cond_lat, img_emb, time_ids = self._cond(cond_image, T)
        sigma = self._sample_sigma()
        with torch.no_grad():
            noise = torch.randn_like(x0)
            z = x0 + sigma * noise
            eps = self._eps_pred(z, sigma, cond_lat, img_emb, time_ids)
            grad = (sigma ** w_power) * (eps - noise)
        return self._apply(x0, grad)

    # --- Motion Distillation Sampling (DreamPhysics, AAAI'25) -------------- #
    def mds_loss(self, frames, cond_image, w_power=0.0):
        """MDS: grad = w(sigma) * (eps_pred(video) - eps_pred(static frame-0)).

        Differencing against the static frame-0 clip cancels the per-frame
        appearance bias, so the gradient carries *motion* only — the fix
        DreamPhysics introduced for SVD's weak motion signal. Same sigma and
        noise are used for both clips so the difference is pure model response.
        """
        T = frames.shape[0]
        x0 = self.encode_frames(frames)                         # [1,T,4,h,w] grad
        with torch.no_grad():
            static = frames[0:1].expand(T, -1, -1, -1)          # frame-0 repeated
            x0_static = self.encode_frames(static)
            cond_lat, img_emb, time_ids = self._cond(cond_image, T)
            sigma = self._sample_sigma()
            noise = torch.randn_like(x0)
            eps_dyn = self._eps_pred(x0 + sigma * noise, sigma, cond_lat, img_emb, time_ids)
            eps_stat = self._eps_pred(x0_static + sigma * noise, sigma, cond_lat, img_emb, time_ids)
            grad = (sigma ** w_power) * (eps_dyn - eps_stat)
        return self._apply(x0, grad)
