"""GIF face-swap job — FaceFusion-class pipeline (YOLO → ArcFace → InSwapper → optional GPEN).

Per selected GIF frame: detect → pair → face-only swap (soft seam blend; optional face-masked
LAB colour match off by default) → optional enhancer → encode verified GIF89a.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import core, detect, gifio, vcore
from .engine import Engine
from .models import ENHANCER_HQ, ENHANCER_LIGHT, ModelStore

log = logging.getLogger("gfs")

ENHANCE_SPECS = {"gpen256": ENHANCER_LIGHT, "gpen512": ENHANCER_HQ}
ENHANCE_LABEL = {None: "Off", "gpen256": "Light (GPEN 256)", "gpen512": "HQ (GPEN 512)"}


class Cancelled(Exception):
    pass


@dataclass
class Settings:
    max_short: int = gifio.MAX_SHORT_SIDE
    enhance: Optional[str] = "gpen256"
    rotation: int = 0
    device: str = "auto"
    out_dir: str = ""
    min_confidence: float = 0.55
    min_face_frac: float = 0.03
    same_gender: bool = False
    # Optional face-masked Reinhard LAB (OFF by default — face-only seam blend)
    color_match: bool = False
    # Optional look/reference image for colour match (face ROI only; no neck/body)
    color_ref_path: str = ""
    seamless: bool = False
    temporal_smooth: float = 0.12
    detector: str = "yolo"

    def detect_opts(self):
        from .detect import DetectOpts
        return DetectOpts(min_confidence=self.min_confidence, min_face_frac=self.min_face_frac,
                          detector=self.detector)


@dataclass
class PhotoFaces:
    path: str
    img: np.ndarray
    faces: list
    hits: list = field(default_factory=list)
    genders: list = field(default_factory=list)


def default_out_dir() -> Path:
    if os.name == "nt":
        try:
            import ctypes, uuid
            fid = uuid.UUID("{33E28130-4E1E-4676-835A-98395C3BC3BB}")  # Pictures
            guid = (ctypes.c_byte * 16).from_buffer_copy(fid.bytes_le)
            pth = ctypes.c_wchar_p()
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(pth)) == 0:
                base = Path(pth.value)
                ctypes.windll.ole32.CoTaskMemFree(pth)
                return base / "GifFaceSwap"
        except Exception:  # noqa: BLE001
            pass
    return Path.home() / "Pictures" / "GifFaceSwap"


default_pictures_dir = default_out_dir


def load_photo(path, opts=None, with_gender=True, engine=None, store=None, device="cpu") -> PhotoFaces:
    from .engine import Engine
    from .models import ModelStore
    img = detect.load_image(path)
    eng = engine
    owned = False
    if eng is None:
        st = store or ModelStore()
        eng = Engine(st, device)
        eng.prepare(None, detector=(opts.detector if opts else "yolo"))
        detect.set_default_engine(eng, "yoloface_8n" if (not opts or opts.detector != "retina") else "retinaface_10g")
        owned = True
    try:
        hits = detect.detect_photo(img, opts, engine=eng)
        if with_gender and hits:
            try:
                from . import gender as G
                G.annotate_hits(img, hits)
            except Exception:  # noqa: BLE001
                pass
        faces = [h.kps.copy() for h in hits]
        genders = [h.gender for h in hits]
        return PhotoFaces(str(path), img, faces, hits=hits, genders=genders)
    finally:
        if owned:
            eng.close()


def load_color_ref(path: str | None) -> Optional[np.ndarray]:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return detect.load_image(p)


class Job:
    def __init__(self, store: ModelStore, device="auto"):
        self.store = store
        self.device = device
        self.engine: Optional[Engine] = None
        self.bench: dict = {}

    def get_engine(self, device=None) -> Engine:
        device = device or self.device
        if self.engine is None or self.engine.device != device:
            if self.engine:
                self.engine.close()
            self.engine = Engine(self.store, device)
            self.device = device
        return self.engine

    def benchmark(self, device=None):
        eng = self.get_engine(device)
        eng.prepare(None)
        self.bench = eng.benchmark()
        return self.bench

    def recommended_enhance(self):
        have = {m for m, s in ENHANCE_SPECS.items() if self.store.is_installed(s)}
        b = self.bench
        if "gpen512" in have and "gpen512" in b and b.get("swap", 1) + b.get("gpen512", 1) <= 0.35:
            return "gpen512"
        if "gpen256" in have:
            return "gpen256"
        return "gpen512" if "gpen512" in have else None

    def _latents(self, eng, photo: PhotoFaces, assign):
        used = sorted({a for a in assign if a >= 0})
        out = {}
        for a in used:
            emb = core.embedding(eng, photo.img, core.kps5(photo.faces[a]))
            out[a] = core.latent_for(eng, emb)
        return out

    def _tick(self, progress, stage, done, total, t0, extra=None):
        if not progress:
            return
        elapsed = max(time.perf_counter() - t0, 1e-3)
        rate = done / elapsed if done else 0.0
        eta = (total - done) / rate if rate > 0 else None
        d = dict(stage=stage, done=done, total=total, rate=rate, eta=eta)
        if extra:
            d.update(extra)
        progress(d)

    def run_gif(self, gif_path, source_photo: PhotoFaces, st: Settings,
                out_path=None, progress=None, cancel=None):
        if not source_photo.faces:
            raise ValueError("No face found in the source faces photo.")
        if st.enhance and st.enhance in ENHANCE_SPECS and not self.store.is_installed(ENHANCE_SPECS[st.enhance]):
            raise ValueError(f"The {ENHANCE_LABEL.get(st.enhance, st.enhance)} enhancer isn't downloaded yet.")

        t_start = time.perf_counter()
        if progress:
            progress(dict(stage="decode", done=0, total=1, detail="Reading GIF…"))
        info, raw_frames = gifio.decode(gif_path)
        if cancel and cancel():
            raise Cancelled()
        W, H, frames = gifio.plan_frames(info, raw_frames, st.max_short)
        n = len(frames)
        if n < 1:
            raise ValueError("No frames to process after applying GIF limits.")
        if progress:
            progress(dict(stage="decode", done=1, total=1,
                          detail=f"{n} frames @ {W}×{H} (from {info.frame_count})"))

        eng = self.get_engine(st.device)
        dev = eng.prepare(st.enhance, detector=st.detector)
        detect.set_default_engine(eng, "yoloface_8n" if st.detector != "retina" else "retinaface_10g")

        color_ref = load_color_ref(st.color_ref_path)
        if color_ref is not None:
            # Resize colour-ref to processing size for ROI alignment
            color_ref = cv2.resize(color_ref, (W, H), interpolation=cv2.INTER_AREA)

        # Detect + pair on first frame with faces to build assign + latents
        if progress:
            progress(dict(stage="detect", done=0, total=n, detail="Finding faces…"))
        t0 = time.perf_counter()
        assign = None
        dets0 = None
        for i, fr in enumerate(frames):
            if cancel and cancel():
                raise Cancelled()
            hits = detect.detect_frame_hits(fr.bgr, max(len(source_photo.faces), 2),
                                            st.detect_opts(), engine=eng)
            dets = [h.kps for h in hits]
            if dets:
                assign = vcore.pair_single_frame(dets, len(source_photo.faces), st.rotation)
                if max(assign) >= 0:
                    dets0 = dets
                    break
            self._tick(progress, "detect", i + 1, n, t0)
        if assign is None or dets0 is None or max(assign) < 0:
            raise ValueError("No face found in that GIF. Try another clip or a clearer face.")

        lat = self._latents(eng, source_photo, assign)
        self._tick(progress, "detect", n, n, t0, {"detail": f"Mapped {sum(1 for a in assign if a >= 0)} face(s)"})

        out_frames: list[tuple[np.ndarray, int]] = []
        t1 = time.perf_counter()
        prev = None
        unchanged = 0
        for i, fr in enumerate(frames):
            if cancel and cancel():
                raise Cancelled()
            hits = detect.detect_frame_hits(fr.bgr, max(len(source_photo.faces), 2),
                                            st.detect_opts(), engine=eng)
            dets = [h.kps for h in hits]
            if not dets:
                out_frames.append((fr.bgr.copy(), fr.delay_ms))
                unchanged += 1
                self._tick(progress, "swap", i + 1, n, t1, {"thumb": fr.bgr, "before": fr.bgr})
                continue
            # Keep left→right pairing stable with source count / rotation
            asg = vcore.pair_single_frame(dets, len(source_photo.faces), st.rotation)
            faces = [(dets[j].astype(np.float32), lat[a]) for j, a in enumerate(asg) if a >= 0 and a in lat]
            if not faces:
                out_frames.append((fr.bgr.copy(), fr.delay_ms))
                unchanged += 1
                self._tick(progress, "swap", i + 1, n, t1)
                continue
            out = core.process_frame(
                eng, fr.bgr, faces, st.enhance,
                color_match=st.color_match, seamless=st.seamless,
                temporal_ema=st.temporal_smooth, prev_out=prev,
                color_ref=color_ref,
            )
            prev = out
            out_frames.append((out, fr.delay_ms))
            extra = {}
            if i % 2 == 0 or i + 1 == n:
                extra = {"thumb": out, "before": fr.bgr}
            self._tick(progress, "swap", i + 1, n, t1, extra)

        if progress:
            progress(dict(stage="encode", done=0, total=1, detail="Writing animated GIF…"))
        out_dir = Path(st.out_dir) if st.out_dir else default_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_path is None:
            stem = "".join(c for c in Path(gif_path).stem if c.isalnum() or c in "-_ ")[:40].strip() or "gif"
            out_path = out_dir / f"GifFaceSwap_{stem}_{time.strftime('%Y%m%d-%H%M%S')}.gif"
        out_path = Path(out_path)
        gifio.encode(out_frames, out_path, loop=True)
        gifio.verify_gif_file(out_path)

        res = dict(
            path=str(out_path), frames=n, W=W, H=H, mode="gif",
            fps=0.0, duration=sum(d for _, d in out_frames) / 1000.0,
            encoder="gif89a", audio="gif (no sound)",
            device=dev.label(), device_active=dev.active, per_model=dict(dev.per_model),
            enhance=ENHANCE_LABEL.get(st.enhance),
            color_match=bool(st.color_match),
            color_ref=bool(color_ref is not None),
            unchanged_frames=unchanged,
            assign=assign, src_faces=len(source_photo.faces),
            detect_s=round(t1 - t0, 2),
            swap_s=round(time.perf_counter() - t1, 2),
            total_s=round(time.perf_counter() - t_start, 2),
            size=out_path.stat().st_size,
            before=frames[0].bgr, after=out_frames[0][0],
        )
        if progress:
            progress(dict(stage="done", done=n, total=n, result=res))
        return res

    # Alias used by some CLI/GUI paths
    run = run_gif
