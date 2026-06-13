"""Benchmark Real-ESRGAN variants on M2 MPS — find fastest config with acceptable quality.

Tests:
  - Real-ESRGAN_x4plus (16.7M params, RRDBNet, current baseline)
  - RealESR-General-x4v3 (2.6M params, SRVGGNetCompact, official lightweight real-photo)

Settings:
  - fp32 vs fp16
  - tile 400 vs 640

Test image: ~/Downloads/u1.jpeg (1080x1440)
"""
import sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from inference import RRDBNet  # 已 inline


# ─── SRVGGNetCompact (for realesr-general-x4v3) ──────────────────
class SRVGGNetCompact(nn.Module):
    """Compact VGG-style net for real-time SR. Used by realesr-general-x4v3."""
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


W = Path(__file__).parent / "weights"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}, torch {torch.__version__}\n")


def load(arch: str, dtype: torch.dtype):
    if arch == "x4plus":
        m = RRDBNet(3, 3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
        state = torch.load(str(W / "RealESRGAN_x4plus.pth"), map_location='cpu', weights_only=True)
        state = state.get('params_ema', state.get('params', state))
    elif arch == "general-x4v3":
        m = SRVGGNetCompact(3, 3, num_feat=64, num_conv=32, upscale=4)
        state = torch.load(str(W / "realesr-general-x4v3.pth"), map_location='cpu', weights_only=True)
        state = state.get('params_ema', state.get('params', state))
    else:
        raise ValueError(f"Unknown arch {arch}")
    m.load_state_dict(state, strict=True)
    m = m.to(DEVICE).eval()
    if dtype == torch.float16:
        m = m.half()
    return m


@torch.no_grad()
def run_tiled(model, img_bgr, dtype, tile=400, overlap=16):
    h, w = img_bgr.shape[:2]
    sf = 4
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)

    out = np.zeros((h*sf, w*sf, 3), dtype=np.float32)
    weight = np.zeros((h*sf, w*sf, 1), dtype=np.float32)
    nx = max(1, (w + tile - 1) // tile)
    ny = max(1, (h + tile - 1) // tile)

    for iy in range(ny):
        for ix in range(nx):
            x0, y0 = ix*tile, iy*tile
            x1 = min(x0 + tile + overlap, w)
            y1 = min(y0 + tile + overlap, h)
            x0_in = max(0, x0 - overlap); y0_in = max(0, y0 - overlap)
            patch = img_t[:, :, y0_in:y1, x0_in:x1].to(DEVICE)
            if dtype == torch.float16:
                patch = patch.half()
            up = model(patch)
            if DEVICE == "mps":
                torch.mps.synchronize()
            up = up.squeeze(0).clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
            oy0, ox0 = y0_in*sf, x0_in*sf
            oy1, ox1 = y1*sf, x1*sf
            out[oy0:oy1, ox0:ox1] += up
            weight[oy0:oy1, ox0:ox1] += 1.0

    out /= np.maximum(weight, 1e-6)
    return cv2.cvtColor((out.clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def bench(arch, dtype, tile, img, label):
    print(f"━━━ {label} ━━━")
    print(f"  arch={arch}, dtype={dtype}, tile={tile}")
    try:
        # Warm up MPS to avoid first-run penalty
        if DEVICE == "mps":
            torch.mps.empty_cache()

        t0 = time.perf_counter()
        m = load(arch, dtype)
        t_load = time.perf_counter() - t0

        # Warmup pass on a tiny image
        warm = np.zeros((64, 64, 3), dtype=np.uint8)
        run_tiled(m, warm, dtype, tile=128)

        t0 = time.perf_counter()
        out = run_tiled(m, img, dtype, tile=tile)
        t_infer = time.perf_counter() - t0

        ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        kb = len(buf) // 1024

        # Save for visual inspection
        out_path = Path.home() / "Downloads" / f"bench_{label.replace(' ', '_').replace('+', '_')}.jpg"
        Path(out_path).write_bytes(buf)

        # Quick numeric sanity (no NaN, valid range)
        if np.isnan(out).any():
            print(f"  ⚠️  NaN in output!")
            quality = "BROKEN"
        elif out.min() == out.max():
            print(f"  ⚠️  Constant output (all {out.min()})")
            quality = "BROKEN"
        else:
            quality = "OK"

        print(f"  Load:     {t_load:.1f}s")
        print(f"  Inference: {t_infer:.2f}s")
        print(f"  Output:   {out.shape[1]}×{out.shape[0]}, {kb} KB JPEG")
        print(f"  Quality:  {quality}")
        print(f"  Saved:    {out_path}\n")
        return {
            "label": label, "arch": arch, "dtype": str(dtype),
            "tile": tile, "load_s": round(t_load, 1),
            "infer_s": round(t_infer, 2), "kb": kb,
            "quality": quality, "path": str(out_path),
        }
    except Exception as e:
        print(f"  ❌ FAILED: {type(e).__name__}: {e}\n")
        return {"label": label, "error": str(e)}


if __name__ == "__main__":
    img = cv2.imread("/Users/wuhonchi/Downloads/u1.jpeg")
    h, w = img.shape[:2]
    print(f"Input: u1.jpeg {w}×{h} ({w*h/1e6:.2f} MP)\n")

    runs = [
        ("x4plus",       torch.float32, 400, "BASELINE x4plus fp32 tile400"),
        ("x4plus",       torch.float16, 400, "x4plus fp16 tile400"),
        ("x4plus",       torch.float32, 640, "x4plus fp32 tile640"),
        ("general-x4v3", torch.float32, 400, "GENERAL-x4v3 fp32 tile400"),
        ("general-x4v3", torch.float16, 400, "GENERAL-x4v3 fp16 tile400"),
        ("general-x4v3", torch.float32, 800, "GENERAL-x4v3 fp32 tile800"),
        ("general-x4v3", torch.float16, 800, "GENERAL-x4v3 fp16 tile800"),
    ]

    results = []
    for arch, dt, tile, label in runs:
        results.append(bench(arch, dt, tile, img, label))

    print("\n" + "="*70)
    print(f"{'Config':45s}  {'Time':>9s}  {'Size':>7s}  Q")
    print("-"*70)
    for r in results:
        if 'error' in r:
            print(f"{r['label']:45s}  {'FAIL':>9s}  {'-':>7s}  ❌")
        else:
            print(f"{r['label']:45s}  {r['infer_s']:>7.2f}s  {r['kb']:>5d}KB  "
                  f"{'✅' if r['quality']=='OK' else '❌'}")
    print("="*70)
    print(f"\nVisual comparison: open ~/Downloads/bench_*.jpg in Preview")
