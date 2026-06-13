"""Real-ESRGAN inference for Apple Silicon MPS.

Default model: RealESR-General-x4v3 (SRVGGNetCompact, 2.6M params).
  - 18× faster than RRDBNet x4plus on M2 MPS (2.4s vs 43s for 1080×1440)
  - Trained on real-world photos (xinntao official)
  - Better fit for Xianyu seller photos than DIV2K-trained models

Inline arch definitions — avoids `basicsr` (broken on Python 3.13).
Architecture code: BSD 3-Clause (xinntao/Real-ESRGAN).
"""
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_DIR = Path(__file__).parent / "weights"

# ─── RRDBNet (Real-ESRGAN x4plus, fallback for quality mode) ─────
def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    h, w = hh // scale, hw // scale
    return x.view(b, c, h, scale, w, scale).permute(0, 1, 3, 5, 2, 4).reshape(b, c * (scale**2), h, w)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3*num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4*num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
    def forward(self, x):
        out = self.rdb1(x); out = self.rdb2(out); out = self.rdb3(out)
        return out * 0.2 + x


def _make_layer(blk, n, **kw):
    return nn.Sequential(*[blk(**kw) for _ in range(n)])


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:   num_in_ch *= 4
        elif scale == 1: num_in_ch *= 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = _make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        if   self.scale == 2: feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1: feat = pixel_unshuffle(x, scale=4)
        else: feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ─── SRVGGNetCompact (RealESR-General-x4v3, DEFAULT — 2.6M params) ──
class SRVGGNetCompact(nn.Module):
    """Compact VGG-style net. Real photo-trained, much faster than RRDBNet."""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
                 upscale=4, act_type='prelu'):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)
    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        return out + base


# ─── Config (defaults set to fastest measured combo) ─────────────
MODEL_NAME = "general-x4v3"     # "general-x4v3" or "x4plus"
USE_FP16 = True                 # 1.13-1.29× speedup on MPS
TILE_DEFAULT = 800              # bigger tile = less Python overhead

_MODEL = None
_DEVICE = None
_DTYPE = None


def get_device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    return _DEVICE


def get_dtype():
    global _DTYPE
    if _DTYPE is None:
        _DTYPE = torch.float16 if USE_FP16 else torch.float32
    return _DTYPE


def load_model():
    global _MODEL
    if _MODEL is None:
        device = get_device()
        dtype = get_dtype()
        if MODEL_NAME == "general-x4v3":
            m = SRVGGNetCompact(3, 3, num_feat=64, num_conv=32, upscale=4)
            weights = WEIGHTS_DIR / "realesr-general-x4v3.pth"
        elif MODEL_NAME == "x4plus":
            m = RRDBNet(3, 3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
            weights = WEIGHTS_DIR / "RealESRGAN_x4plus.pth"
        else:
            raise ValueError(f"Unknown MODEL_NAME={MODEL_NAME}")
        state = torch.load(str(weights), map_location='cpu', weights_only=True)
        state = state.get('params_ema', state.get('params', state))
        m.load_state_dict(state, strict=True)
        m = m.to(device).eval()
        if dtype == torch.float16:
            m = m.half()
        _MODEL = m
    return _MODEL


@torch.no_grad()
def upscale_tiled(img_bgr_uint8: np.ndarray, tile: int = TILE_DEFAULT, overlap: int = 16,
                  progress=None) -> np.ndarray:
    model = load_model()
    device = get_device()
    dtype = get_dtype()
    h, w = img_bgr_uint8.shape[:2]
    sf = 4

    img_rgb = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)

    out = np.zeros((h*sf, w*sf, 3), dtype=np.float32)
    weight = np.zeros((h*sf, w*sf, 1), dtype=np.float32)
    nx = max(1, (w + tile - 1) // tile)
    ny = max(1, (h + tile - 1) // tile)
    total = nx * ny
    n = 0

    for iy in range(ny):
        for ix in range(nx):
            x0, y0 = ix*tile, iy*tile
            x1 = min(x0 + tile + overlap, w)
            y1 = min(y0 + tile + overlap, h)
            x0_in = max(0, x0 - overlap); y0_in = max(0, y0 - overlap)
            patch = img_t[:, :, y0_in:y1, x0_in:x1].to(device)
            if dtype == torch.float16:
                patch = patch.half()
            up = model(patch)
            if device == "mps":
                torch.mps.synchronize()
            up = up.squeeze(0).clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
            oy0, ox0 = y0_in*sf, x0_in*sf
            oy1, ox1 = y1*sf, x1*sf
            out[oy0:oy1, ox0:ox1] += up
            weight[oy0:oy1, ox0:ox1] += 1.0
            n += 1
            if progress: progress(n, total)

    out /= np.maximum(weight, 1e-6)
    return cv2.cvtColor((out.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def pick_tile(max_edge: int) -> int:
    # SRVGGNetCompact is light — bigger tiles are fine
    if max_edge <= 1000: return TILE_DEFAULT
    if max_edge <= 2000: return 600
    return 400


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "/Users/wuhonchi/Downloads/u1.jpeg"
    img = cv2.imread(inp)
    if img is None:
        print(f"ERROR: cannot read {inp}"); sys.exit(1)
    h, w = img.shape[:2]
    tile = pick_tile(max(h, w))
    print(f"Input: {w}×{h}  device={get_device()} dtype={get_dtype()} model={MODEL_NAME} tile={tile}")
    load_model()
    t0 = time.perf_counter()
    out = upscale_tiled(img, tile=tile)
    elapsed = time.perf_counter() - t0
    p = Path(inp).with_name(Path(inp).stem + "_x4_local.jpg")
    cv2.imwrite(str(p), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"✓ {elapsed:.2f}s → {p}")
