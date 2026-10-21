"""Image encoders for the wrist RGB + depth ablation (task configs stack_wrist_rgbd / _depth_late /
_depth_xattn). Plugged into diffusion_policy's DiffusionUnetImagePolicy through the config's
obs_encoder / rgb_model targets; the repo itself is unmodified.

  get_resnet_in_chans   torchvision ResNet with conv1 for any number of input channels
                        (4 for RGB-D early fusion, 1 for a depth-only trunk); random init like the
                        paper's encoders. diffusion_policy's MultiImageObsEncoder then replaces its
                        BatchNorm with GroupNorm as for RGB.
  RGBDCrossAttnEncoder  variant C: RGB and depth trunks -> spatial tokens -> Perceiver-style latent
                        cross-attention fusion. Same interface as MultiImageObsEncoder
                        (forward(obs_dict) on single-timestep batches -> (N, D); output_shape()).
"""

from typing import Dict

import torch
import torch.nn as nn
import torchvision

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


def get_resnet_in_chans(name="resnet18", in_chans=3, weights=None):
    """ResNet (torchvision) taking `in_chans` input channels, fc removed (512-D pooled output for
    resnet18/34), like diffusion_policy's get_resnet for 3 channels."""
    resnet = getattr(torchvision.models, name)(weights=weights)
    resnet.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Identity()
    return resnet


def _group_norm(module):
    """BatchNorm -> GroupNorm (16 channels per group), as MultiImageObsEncoder does: EMA weights and
    BatchNorm statistics don't mix (paper Sec. 3.2)."""
    return replace_submodules(
        root_module=module, predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features))


def _trunk(name, in_chans):
    """ResNet up to layer4, no global pool: (N, C, H, W) -> (N, 512, H/32, W/32) for resnet18."""
    r = get_resnet_in_chans(name, in_chans)
    return _group_norm(nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4))


def sincos_2d(h, w, dim, device=None):
    """(h*w, dim) fixed 2D sine-cosine positional embedding (half the channels for y, half for x)."""
    assert dim % 4 == 0
    d = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, dtype=torch.float32, device=device) / d))
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device),
                          torch.arange(w, dtype=torch.float32, device=device), indexing="ij")
    oy, ox = y.reshape(-1, 1) * omega, x.reshape(-1, 1) * omega
    return torch.cat([oy.sin(), oy.cos(), ox.sin(), ox.cos()], dim=1)


class _FusionBlock(nn.Module):
    """latents <- cross-attention to the tokens, then latent self-attention, then MLP (pre-LN)."""

    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.ln_q, self.ln_kv = nn.LayerNorm(d), nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln_s = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln_m = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))

    def forward(self, lat, tok):
        kv = self.ln_kv(tok)
        lat = lat + self.cross(self.ln_q(lat), kv, kv, need_weights=False)[0]
        s = self.ln_s(lat)
        lat = lat + self.self_attn(s, s, s, need_weights=False)[0]
        return lat + self.mlp(self.ln_m(lat))


class RGBDCrossAttnEncoder(ModuleAttrMixin):
    """Variant C: Perceiver-style cross-attention fusion of wrist RGB and depth.

    1. One random crop window per sample, applied to BOTH images (pixel alignment kept);
       centre crop at inference.
    2. Separate ResNet trunks (3-ch RGB, 1-ch depth) up to layer4, no pooling:
       216x288 -> 7x9 = 63 spatial tokens per modality, projected (1x1 conv) to d_model.
    3. + a fixed 2D sine-cosine position embedding (the same grid for both modalities, since
       they are pixel-aligned) + a learned modality embedding.
    4. n_latents learned queries; n_layers x [cross-attention to all 2*63 tokens -> latent
       self-attention -> MLP].
    5. Flattened latents -> Linear -> out_dim, concatenated with the low_dim keys (proprio), as
       MultiImageObsEncoder concatenates its features.
    """

    def __init__(self, shape_meta: dict, rgb_key="wrist", depth_key="wrist_depth", backbone="resnet18",
                 crop_shape=(216, 288), random_crop=True, d_model=256, n_latents=16, n_layers=2, n_heads=8,
                 out_dim=512, dropout=0.0):
        super().__init__()
        obs_meta = shape_meta["obs"]
        image_keys = {k for k, v in obs_meta.items() if v.get("type", "low_dim") == "rgb"}
        if image_keys != {rgb_key, depth_key}:
            raise ValueError(f"RGBDCrossAttnEncoder needs exactly the image keys {{{rgb_key}, {depth_key}}}, "
                             f"got {sorted(image_keys)}")
        self.shape_meta = shape_meta
        self.rgb_key, self.depth_key = rgb_key, depth_key
        self.low_dim_keys = sorted(k for k, v in obs_meta.items() if v.get("type", "low_dim") == "low_dim")
        self.crop_shape = tuple(crop_shape) if crop_shape is not None else None
        self.random_crop = random_crop
        self.rgb_trunk = _trunk(backbone, tuple(obs_meta[rgb_key]["shape"])[0])
        self.depth_trunk = _trunk(backbone, tuple(obs_meta[depth_key]["shape"])[0])
        c = 512 if backbone in ("resnet18", "resnet34") else 2048
        self.proj_rgb = nn.Conv2d(c, d_model, kernel_size=1)
        self.proj_depth = nn.Conv2d(c, d_model, kernel_size=1)
        self.modality = nn.Parameter(torch.zeros(2, d_model))
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.blocks = nn.ModuleList(_FusionBlock(d_model, n_heads, dropout) for _ in range(n_layers))
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(n_latents * d_model, out_dim)
        nn.init.normal_(self.modality, std=0.02)

    def _crop(self, rgb, depth):
        if self.crop_shape is None:
            return rgb, depth
        ch, cw = self.crop_shape
        H, W = rgb.shape[-2:]
        if self.training and self.random_crop:
            n = rgb.shape[0]
            ys = torch.randint(0, H - ch + 1, (n,)).tolist()
            xs = torch.randint(0, W - cw + 1, (n,)).tolist()
            rgb = torch.stack([rgb[i, :, y:y + ch, x:x + cw] for i, (y, x) in enumerate(zip(ys, xs))])
            depth = torch.stack([depth[i, :, y:y + ch, x:x + cw] for i, (y, x) in enumerate(zip(ys, xs))])
            return rgb, depth
        y, x = (H - ch) // 2, (W - cw) // 2
        return rgb[..., y:y + ch, x:x + cw], depth[..., y:y + ch, x:x + cw]

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        rgb, depth = self._crop(obs_dict[self.rgb_key], obs_dict[self.depth_key])
        f_rgb = self.proj_rgb(self.rgb_trunk(rgb))            # (N, d, h, w)
        f_depth = self.proj_depth(self.depth_trunk(depth))
        n, d, h, w = f_rgb.shape
        pos = sincos_2d(h, w, d, device=f_rgb.device).to(f_rgb.dtype)
        tok = torch.cat([f_rgb.flatten(2).transpose(1, 2) + pos + self.modality[0],
                         f_depth.flatten(2).transpose(1, 2) + pos + self.modality[1]], dim=1)   # (N, 2hw, d)
        lat = self.latents.unsqueeze(0).expand(n, -1, -1)
        for blk in self.blocks:
            lat = blk(lat, tok)
        feat = self.head(self.ln_out(lat).flatten(1))
        lowdim = [obs_dict[k].reshape(n, -1) for k in self.low_dim_keys]
        return torch.cat([feat] + lowdim, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        example = {k: torch.zeros((1,) + tuple(v["shape"]), dtype=self.dtype, device=self.device)
                   for k, v in self.shape_meta["obs"].items()}
        return self.forward(example).shape[1:]
