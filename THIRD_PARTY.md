# Third-party notices — GIF Face Swap for Windows

| Component | Licence | Notes |
|---|---|---|
| **ONNX Runtime DirectML** | MIT | Microsoft. GPU EP for Radeon 780M. |
| **OpenCV** (headless) | Apache-2.0 | |
| **NumPy** | BSD | |
| **Pillow** | HPND | GIF decode. |
| **PySide6 / Qt 6** | LGPLv3 | Qt Company. Dynamic link via PySide wheels. |
| **PyInstaller** | GPLv2+ with exception / Apache-2.0 bootloader | Build-time only. |
| **YOLO Face (`yoloface_8n`)** | GPL-3.0 | derronqi; distributed via FaceFusion assets. |
| **InsightFace ArcFace / inswapper / RetinaFace / genderage** | Non-commercial research | insightface.ai — commercial use needs a separate licence. |
| **GPEN-BFR** | Research (Alibaba DAMO) | Treat as personal / non-commercial. |
| **GFPGAN** (optional) | Apache-2.0 | Tencent ARC. |
| **FaceFusion model hosting** | Various | Models downloaded from `facefusion/facefusion-assets` releases / HF mirror. This app is **not** a redistribution of the FaceFusion GPL application. |

GIF encode: in-process GIF89a (median-cut + LZW). Lessons from Android GifFaceSwap 1.0.1 (Clear at current code width + Global Color Table).

Test image: see `testdata/SOURCES.md`. Private people / CC0 only — no public figures.
