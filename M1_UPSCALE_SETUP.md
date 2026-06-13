# CardJeng Upscale — M1 Mac mini setup

This guide bootstraps the self-hosted Real-ESRGAN upscale service on an M1 Mac
mini and wires it back into `cardjeng.pages.dev`. Designed so Claude Code on the
M1 can read this file and execute the steps end-to-end with minimal user input.

Target: a long-lived 24/7 host that the public CardJeng page calls when a user
taps **✨ AI Upscale**.

---

## 0. Architecture (what we are building)

```
User (any browser, anywhere)
  │  POST card photo
  ▼
https://cardjeng.pages.dev/             ◄── static site (Cloudflare Pages,
  │   index.html → "✨ AI Upscale"           git-deployed from main branch)
  │
  ▼   fetch(UPSCALE_URL_DEFAULT)
https://xxxxx.trycloudflare.com/        ◄── Quick Tunnel, free, URL rotates
  │                                          on cloudflared restart
  ▼
cloudflared daemon (on M1)
  │
  ▼
http://127.0.0.1:8000                   ◄── FastAPI + uvicorn worker
  │   POST /jobs   →  job_id queued
  │   GET  /jobs/<id>           polled
  │   GET  /jobs/<id>/result    JPEG
  ▼
PyTorch MPS + Real-ESRGAN              ◄── ~3 s / 1080×1440 → 4320×5760 on M2
  (realesr-general-x4v3 default)            (M1 estimate: 4-6 s, unmeasured)
```

The server is **anonymous** by default — there is no bearer token. The only
abuse safeguard is a per-IP rate limit of 60 jobs/hour. The tunnel URL is
hardcoded in `index.html`; anyone viewing the page source can see it. For
personal solo use this is the accepted trade-off.

---

## 1. Prerequisites on the M1 mini

```bash
# macOS 13+ is fine; needs Homebrew + Python 3.10+.
brew --version >/dev/null 2>&1 || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
python3 --version    # expect 3.10 or newer
xcode-select --install   # only if `clang` isn't present
```

Then make sure the repo is checked out:

```bash
mkdir -p ~/Documents && cd ~/Documents
git clone https://github.com/wuhonchi/poke-lookup.git 2>/dev/null \
  || (cd poke-lookup && git pull)
cd poke-lookup
```

---

## 2. Install cloudflared + Python venv + deps

```bash
brew install cloudflared

mkdir -p ~/cardjeng-tools/{weights,logs,jobs}
python3 -m venv ~/cardjeng-tools/venv

~/cardjeng-tools/venv/bin/pip install --quiet --upgrade pip
~/cardjeng-tools/venv/bin/pip install --quiet \
  torch torchvision \
  fastapi 'uvicorn[standard]' \
  Pillow opencv-python python-multipart
```

Verify MPS is detected:

```bash
~/cardjeng-tools/venv/bin/python -c "
import torch
print('torch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
"
```

Expected:
```
torch: 2.x.x
MPS available: True
```

---

## 3. Download Real-ESRGAN weights

```bash
W=~/cardjeng-tools/weights

# Primary model (default — fastest, real-photo trained)
[ -f "$W/realesr-general-x4v3.pth" ] || curl -sL -o "$W/realesr-general-x4v3.pth" \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"

# Optional: heavier quality fallback (16.7M-param model)
[ -f "$W/RealESRGAN_x4plus.pth" ] || curl -sL -o "$W/RealESRGAN_x4plus.pth" \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

ls -lh "$W/"
# Expect ~4.7 MB + ~64 MB
```

---

## 4. Copy server code into `~/cardjeng-tools/`

The Python source lives in this repo at `scripts/upscale-server/`. Copy it to
the runtime location:

```bash
cp scripts/upscale-server/inference.py ~/cardjeng-tools/
cp scripts/upscale-server/server.py    ~/cardjeng-tools/
cp scripts/upscale-server/bench.py     ~/cardjeng-tools/
```

If you want to switch the default model later, edit `~/cardjeng-tools/inference.py`:

```python
# Top of inference.py
MODEL_NAME = "general-x4v3"  # or "x4plus" for heavier quality
USE_FP16 = True              # ~10 % speedup on MPS
TILE_DEFAULT = 800
```

---

## 5. Quick local sanity check (no tunnel yet)

Run inference directly on a test image:

```bash
~/cardjeng-tools/venv/bin/python ~/cardjeng-tools/inference.py /path/to/any.jpg
# Saves /path/to/any_x4_local.jpg
```

Expected output:

```
Input: WxH device=mps dtype=torch.float16 model=general-x4v3 tile=800
✓ N.NNs → /path/to/any_x4_local.jpg
```

If `device=cpu` shows up, MPS isn't available — re-check torch install and
that the M1 is running natively (`uname -m` returns `arm64`, not `x86_64`).

---

## 6. Start the FastAPI server

Anonymous mode (matches current production setup):

```bash
# Kill any previous instance
[ -f ~/cardjeng-tools/.server.pid ] && kill -9 $(cat ~/cardjeng-tools/.server.pid) 2>/dev/null

cd ~/cardjeng-tools
nohup ./venv/bin/uvicorn server:app \
  --host 127.0.0.1 --port 8000 --workers 1 \
  > ~/cardjeng-tools/logs/server.log 2>&1 &
echo $! > ~/cardjeng-tools/.server.pid
disown
sleep 4

# Sanity:
curl -s http://127.0.0.1:8000/health
```

Expected:

```json
{"ok":true,"device":"mps","queue_depth":0,"active":false,"jobs":0}
```

If you ever want a bearer-token mode again, export `CARDJENG_TOKEN=somesecret`
before starting uvicorn — the server will require `Authorization: Bearer …`
on every `/jobs*` request and will skip the anonymous rate limit when a valid
token is present.

---

## 7. Start the Cloudflare Quick Tunnel

```bash
pkill -f "cloudflared tunnel --url" 2>/dev/null
sleep 1

nohup cloudflared tunnel --url http://127.0.0.1:8000 \
  > ~/cardjeng-tools/logs/tunnel.log 2>&1 &
echo $! > ~/cardjeng-tools/.tunnel.pid
disown
sleep 10   # cloudflared takes a few seconds to register

# Grab the public URL
TUNNEL_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' \
  ~/cardjeng-tools/logs/tunnel.log | head -1)
echo "Tunnel URL: $TUNNEL_URL"
echo "$TUNNEL_URL" > ~/cardjeng-tools/.tunnel-url

# End-to-end test:
curl -s "$TUNNEL_URL/health"
```

The Quick Tunnel URL is **random and rotates** every time `cloudflared`
restarts. To keep a stable URL long-term, see **section 11 (deferred)**.

---

## 8. Wire CardJeng to the new tunnel URL

CardJeng has a single hardcoded constant pointing at the upscale endpoint:

```text
index.html  (search for UPSCALE_URL_DEFAULT)
  const UPSCALE_URL_DEFAULT = 'https://OLD_HOST.trycloudflare.com';
```

Update it to the new M1 tunnel URL and push:

```bash
cd ~/Documents/poke-lookup

# Replace the URL (BSD sed for macOS — different from GNU sed)
sed -i '' "s|https://[a-z0-9-]*\.trycloudflare\.com|$TUNNEL_URL|" index.html
grep -n UPSCALE_URL_DEFAULT index.html

git add index.html
git commit -m "Upscale: switch to M1 mini tunnel ($TUNNEL_URL)"
git push origin main
```

Cloudflare Pages auto-deploys in ~30-60 s. After that, `cardjeng.pages.dev`
sends ✨ AI Upscale jobs to the M1.

---

## 9. End-to-end verify from a browser

1. Open `https://cardjeng.pages.dev/`
2. Upload any card photo
3. The ✨ **AI Upscale** button appears in the toolbar.
4. Tap it. Expected end-to-end time: 5–10 s on M1.
5. The image on the canvas should be replaced by the upscaled version
   (4× resolution).
6. Tap from an iPhone on the same network too — the request still goes via
   the tunnel, so it works equally from cellular / different Wi-Fi.

If the upscale fails, check:

```bash
tail -f ~/cardjeng-tools/logs/server.log
```

You should see lines like:

```
queued <id> input=1080×1440 ...KB queue=0
running <id> 1080×1440 tile=800
done <id> 4320×5760 ...KB elapsed=...s
```

---

## 10. (Optional) Auto-start on boot via `launchctl`

The M1 mini is meant to host 24/7. Two LaunchAgents make the stack survive a
reboot.

Create `~/Library/LaunchAgents/com.cardjeng.upscale.server.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>com.cardjeng.upscale.server</string>
  <key>WorkingDirectory</key>  <string>/Users/USERNAME/cardjeng-tools</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/USERNAME/cardjeng-tools/venv/bin/uvicorn</string>
    <string>server:app</string>
    <string>--host</string><string>127.0.0.1</string>
    <string>--port</string><string>8000</string>
    <string>--workers</string><string>1</string>
  </array>
  <key>RunAtLoad</key>           <true/>
  <key>KeepAlive</key>           <true/>
  <key>StandardOutPath</key>     <string>/Users/USERNAME/cardjeng-tools/logs/server.log</string>
  <key>StandardErrorPath</key>   <string>/Users/USERNAME/cardjeng-tools/logs/server.log</string>
</dict>
</plist>
```

And `~/Library/LaunchAgents/com.cardjeng.upscale.tunnel.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>            <string>com.cardjeng.upscale.tunnel</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/cloudflared</string>
    <string>tunnel</string>
    <string>--url</string>
    <string>http://127.0.0.1:8000</string>
  </array>
  <key>RunAtLoad</key>          <true/>
  <key>KeepAlive</key>          <true/>
  <key>StandardOutPath</key>    <string>/Users/USERNAME/cardjeng-tools/logs/tunnel.log</string>
  <key>StandardErrorPath</key>  <string>/Users/USERNAME/cardjeng-tools/logs/tunnel.log</string>
</dict>
</plist>
```

Replace `USERNAME` with `whoami` output, then:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cardjeng.upscale.server.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cardjeng.upscale.tunnel.plist
```

After reboot, the server + tunnel come back up automatically. The Quick Tunnel
URL **will be new** — you'll need to repeat **section 8** (or move to a
persistent URL strategy, see section 11).

---

## 11. (Deferred) Persistent tunnel URL

Quick Tunnel rotates URLs on every restart. To stop editing `index.html` after
every M1 reboot, one of:

| Option | Cost | URL shape | Effort |
|--------|------|-----------|--------|
| **Cloudflare named tunnel + own domain** | ~$10/yr | `upscale.your-domain.com` | 30 min, needs domain in Cloudflare |
| **Tailscale Funnel** | $0 personal | `mac.tail-XXXX.ts.net` | 15 min, sign-up + Funnel toggle |
| **Stay on Quick Tunnel** | $0 | random `*.trycloudflare.com` | 0, but edit on every restart |

For now we are on Quick Tunnel. Migrate when reboots become annoying.

---

## 12. Troubleshooting

| Symptom | Diagnosis | Fix |
|--------|-----------|-----|
| Page shows ✨ but click does nothing | Tunnel down | `curl -s $UPSCALE_URL/health` — should return JSON. If not, `pkill -f cloudflared` then re-run section 7. |
| `503 / 524 Cloudflare error` | Tunnel up but server crashed | `tail ~/cardjeng-tools/logs/server.log` — restart per section 6 |
| `429 Queue full` | More than 4 jobs queued | Wait — current job finishes in seconds |
| `429 Rate limit` | Hit anonymous 60/hr/IP limit | Wait an hour or set `CARDJENG_TOKEN` and authenticate |
| `413 Upload exceeds 5 MB` | Source photo too large | Server caps incoming files; user should reduce input |
| `413 Input edge exceeds 3000` | Photo wider/taller than 3000 px | Same — downscale source first |
| Inference crashes "Error: MPS backend" | MPS install broken | `pip install --upgrade --force-reinstall torch torchvision` |
| Inference uses CPU (slow) | M1 not running native arm64 | Re-create venv after running `arch -arm64 brew install python` |
| Browser CORS error | Origin not in allowlist | `server.py` regex covers `localhost / 127.0.0.1 / 192.168.x.x / 10.x.x.x` + `https://cardjeng.pages.dev`. If you self-host the front from a different host, add it to `allow_origin_regex` |
| `health` returns `device:"cpu"` | MPS not detected | Quit + reopen Terminal so PyTorch picks up MPS env |
| Service didn't come back after reboot | LaunchAgent not loaded | `launchctl list | grep cardjeng` — re-bootstrap from section 10 |

---

## 13. Switching from M2 → M1

Once M1 is verified:

1. On the **M2** dev machine, stop the old stack:

   ```bash
   kill $(cat ~/cardjeng-tools/.server.pid) 2>/dev/null
   kill $(cat ~/cardjeng-tools/.tunnel.pid) 2>/dev/null
   ```

   The `~/cardjeng-tools/` dir can stay (handy for future testing).

2. Confirm `index.html` `UPSCALE_URL_DEFAULT` now points at the **M1** tunnel
   URL and that push has deployed to `cardjeng.pages.dev`.

3. Public users keep working — they never knew the URL switched.

---

## 14. State park (deferred decisions)

These were intentionally not done — capture them so future you / Claude don't
re-derive:

- **Persistent URL**: stuck on Quick Tunnel until reboot pain forces Tailscale
  Funnel or buying a domain.
- **Auth**: server runs anonymous + 60/hr/IP rate limit. Bearer token is
  supported (set `CARDJENG_TOKEN`); not enabled because user wanted
  zero-prompt UX. If abuse appears, enable token + hardcode it client-side.
- **`launchctl` autostart**: optional section 10 above. Worth doing on M1
  since the box is the long-term host.
- **CoreML / Neural Engine**: not attempted. Could be 2–3× faster than MPS
  for the same model, but ONNX export + op-by-op verification is ~1 day work.
- **CardJeng integration polish**: in-button progress is text-only. Could be
  replaced with a determinate progress bar, an inline preview thumbnail, etc.
- **Old `/centering` redirect**: handled by `_redirects` in repo root. Don't
  remove it — there may be external bookmarks.

---

## 15. Reference — files this depends on

| Path | Purpose |
|------|---------|
| `scripts/upscale-server/inference.py` | Model definitions + tile-based inference |
| `scripts/upscale-server/server.py`    | FastAPI app, job queue, rate limit, CORS |
| `scripts/upscale-server/bench.py`     | Benchmark across model × dtype × tile |
| `index.html`                          | `UPSCALE_URL_DEFAULT` constant points at tunnel |
| `_redirects`                          | Old `/centering` URL → `/` (301) |

End.
