# GIF Face Swap for Windows

Touch-friendly **Windows x64** app for the **ASUS ROG Ally X** that swaps a face from a photo onto every frame of an animated GIF.

**Powered by FaceFusion-compatible ONNX models** (same quality stack as Face Fusion Studio): YOLO Face detector, ArcFace identity, inswapper_128 (fp16), optional GPEN enhancer, plus **body/skin colour match (Reinhard LAB) on by default** so the swapped face matches the GIF subject’s skin tone (neck/blend region included). Optional separate colour-look reference image in Options.

| | This app |
|---|---|
| Package | GIF Face Swap **1.0.0** (portable zip) |
| Acceleration | **ONNX Runtime DirectML** on Radeon 780M, automatic CPU fallback (child-process GPU probe) |
| Mode | **Animated GIF** face swap |
| Output | `%USERPROFILE%\Pictures\GifFaceSwap\` — verified GIF89a |

**Tech:** Python 3.13 + PySide6 + onnxruntime-directml + OpenCV + Pillow, packaged with PyInstaller on GitHub Actions `windows-latest`. **No FFmpeg** (GIF encode is in-process). **No MediaPipe.**

> Standalone product — **not** an update to Face Fusion Studio, Face Swap Video (Windows), or the Android GIF Face Swap app. Those releases are untouched.

## Download

Grab **`GifFaceSwap-1.0.0-win64.zip`** from [Releases](https://github.com/vanukrishnans-source/gif-face-swap-windows/releases/tag/v1.0.0).

| Asset | Size | SHA-256 |
|---|---|---|
| [GifFaceSwap-1.0.0-win64.zip](https://github.com/vanukrishnans-source/gif-face-swap-windows/releases/download/v1.0.0/GifFaceSwap-1.0.0-win64.zip) | 147,621,132 B (~140.8 MB) | `a0040ddcadbe83568cf60a2c72d20efc0782440a437c1b6090acdb3114f380d7` |

### First launch (SmartScreen)

The build is **not code-signed**. Windows SmartScreen will say “Windows protected your PC”:

1. Click **More info**
2. Click **Run anyway**

```powershell
Get-FileHash .\GifFaceSwap-1.0.0-win64.zip -Algorithm SHA256
```

### Install / run on Ally X

1. Unzip anywhere (e.g. `C:\Games\GifFaceSwap\`).
2. Double-click **`GifFaceSwap.exe`**.
3. First run downloads **~465 MB** required models (YOLO Face 8n + ArcFace + inswapper_128 fp16). Optional: Light GPEN ≈ 76 MB, HQ GPEN ≈ 284 MB.
4. If Face Fusion Studio (or Face Swap Video) already downloaded the same models, they are reused automatically from `%LOCALAPPDATA%\FaceFusionStudio\models`.

Models stay in `%LOCALAPPDATA%\GifFaceSwap\models`.

## How to use

1. Tap **Animated GIF** → pick a `.gif`.
2. Tap **Face from photo** → pick a clear face photo.
3. Optional: **Flip faces**, **Options** (enhancer, colour match, colour-look reference, device).
4. Tap **Create face swap GIF**.
5. Open the result in Photos / browser — it is a valid animated GIF89a.

## Colour match (required feature, on by default)

- Matches the swapped face (and soft chin/neck blend) to the **GIF subject’s body/skin tone** via Reinhard LAB (same idea as Face Fusion Studio’s colour match).
- Options → uncheck to disable.
- Options → **Pick colour reference…** to match a separate look image instead of the GIF frame.

## GIF limits

| Limit | Value |
|---|---|
| Max frames processed | 80 (evenly subsampled if longer) |
| Max duration used | 12 seconds |
| Max short side while processing | 480 px (Options up to 720) |
| Max GIF file size | 25 MB |
| Output | Animated GIF (GIF89a, GCT + LZW Clear-at-current-width) |

Frames with no detectable face are left unchanged.

## Models

| file | bytes |
|---|---|
| yoloface_8n.onnx | 12,659,761 |
| arcface_w600k_r50.onnx | 174,388,474 |
| inswapper_128_fp16.onnx | 277,680,829 |
| gpen_bfr_256.onnx (optional Light) | 75,792,988 |
| gpen_bfr_512.onnx (optional HQ) | 284,340,240 |

InsightFace models: personal / non-commercial research unless you have a commercial licence from insightface.ai.

## Build from source

Needs Windows x64 (or the GitHub Actions workflow).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-win.txt
$env:GFS_MODELS = "$PWD\.models-cache"
python scripts/prefetch_models.py light
python scripts/make_test_gif.py
python -m gfs --selftest --gif testdata\sample_faces.gif --photo testdata\faces_e5.jpg --models $env:GFS_MODELS --device cpu --enhance gpen256 --max-short 320
pyinstaller GifFaceSwap.spec --noconfirm
powershell -File packaging\make_zip.ps1
```

## Related (untouched)

- [face-fusion-windows](https://github.com/vanukrishnans-source/face-fusion-windows) — Face Fusion Studio (photo + video)
- [face-swap-video-windows](https://github.com/vanukrishnans-source/face-swap-video-windows) — Face Swap Video
- [gif-face-swap](https://github.com/vanukrishnans-source/gif-face-swap) — Android GIF Face Swap

## Licence / models

- **App code:** MIT (see `LICENSE`).
- **InsightFace models:** personal / non-commercial research only.
- **YOLO Face:** GPL-3.0 (derronqi).
- **GPEN-BFR:** research / personal.
- Don’t use this app to impersonate or deceive anyone; only swap faces of people who agreed to it.
