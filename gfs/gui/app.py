"""GIF Face Swap — touch-first Qt UI for the ROG Ally X (7\" 1080p @ 150%).

Flow: pick animated GIF → pick face photo → optional Flip / Options → Create.
Face-only soft seam blend by default; optional face-masked LAB colour match is OFF.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QStackedWidget, QVBoxLayout, QWidget,
)

from .. import __version__, detect, gifio, vcore
from ..job import (
    ENHANCE_LABEL, ENHANCE_SPECS, Cancelled, Job, Settings, default_out_dir, load_photo,
)
from ..models import ENHANCER_LIGHT, ENHANCER_HQ, ModelStore, REQUIRED_BYTES
from .theme import DARK_QSS

log = logging.getLogger("gfs")

TEAL = (169, 184, 0)
PAGE_SETUP, PAGE_MAIN, PAGE_OPTIONS, PAGE_PROGRESS, PAGE_DONE = range(5)
GIF_EXT = {".gif"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def pix(bgr, w, h):
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return QPixmap()
    dpr = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
    W, H = int(w * dpr), int(h * dpr)
    ih, iw = bgr.shape[:2]
    if ih < 1 or iw < 1:
        return QPixmap()
    s = min(W / iw, H / ih)
    img = cv2.resize(bgr, (max(1, int(iw * s)), max(1, int(ih * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    p = QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                                 QImage.Format.Format_RGB888).copy())
    p.setDevicePixelRatio(dpr)
    return p


def face_crop(img, f, size=120):
    x0, y0, x1, y1 = vcore.bbox(f)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(x1 - x0, y1 - y0) * 0.7
    a, b = int(max(0, cx - r)), int(max(0, cy - r))
    c, d = int(min(img.shape[1], cx + r)), int(min(img.shape[0], cy + r))
    if d <= b or c <= a:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(img[b:d, a:c], (size, size), interpolation=cv2.INTER_AREA)


class Worker(QThread):
    progressed = Signal(object)
    done = Signal(object)
    failed = Signal(str, str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.cancelled = False

    def run(self):
        try:
            self.done.emit(self.fn(lambda: self.cancelled, self.progressed.emit))
        except Cancelled:
            self.failed.emit("cancelled", "")
        except Exception as e:  # noqa: BLE001
            if self.cancelled or str(e) == "cancelled":
                self.failed.emit("cancelled", "")
            else:
                tb = traceback.format_exc()
                log.error("worker failed: %s\n%s", e, tb)
                try:
                    from .. import crashlog
                    crashlog.record_current(where="worker")
                except Exception:  # noqa: BLE001
                    pass
                self.failed.emit(str(e) or type(e).__name__, tb)


def button(text, kind=None, min_w=0):
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if min_w:
        b.setMinimumWidth(min_w)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setMinimumHeight(56)
    return b


def label(text="", kind=None, wrap=False):
    l = QLabel(text)
    if kind:
        l.setObjectName(kind)
    l.setWordWrap(wrap)
    return l


def card(oid="card"):
    f = QFrame()
    f.setObjectName(oid)
    return f


class ImageSlot(QFrame):
    clicked = Signal()
    dropped = Signal(str)

    def __init__(self, title, hint, h=220):
        super().__init__()
        self.setObjectName("card")
        self.h = h
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        self.title = label(title, "section")
        lay.addWidget(self.title)
        self.image = QLabel(hint)
        self.image.setObjectName("slot")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumHeight(h)
        self.image.setWordWrap(True)
        lay.addWidget(self.image, 1)
        self.info = label("", "hint", True)
        lay.addWidget(self.info)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            self.dropped.emit(u.toLocalFile())

    def set_image(self, bgr):
        self.image.setPixmap(pix(bgr, self.image.width() or 600, self.h))

    def clear_image(self, hint="Tap to choose"):
        self.image.clear()
        self.image.setText(hint)


class MainWindow(QMainWindow):
    def __init__(self, store: ModelStore, device="auto"):
        super().__init__()
        self.setWindowTitle(f"GIF Face Swap {__version__}")
        self.cfg = QSettings("vanu", "GifFaceSwap")
        self.store = store
        dev = str(self.cfg.value("device", device))
        dev = dev if dev in ("auto", "dml", "cpu") else "auto"
        self.job = Job(store, device if device != "auto" else dev)
        self.worker = None
        self.gif_path = None
        self.gif_info = None
        self.gif_preview = None
        self.photo = None
        self.photo_path = None
        self.rotation = 0
        self.result = None
        self.bench_done = False
        # One-shot quality bump: old installs defaulted to 480 short-side (soft/pixelated).
        _ms = int(self.cfg.value("max_short", gifio.MAX_SHORT_SIDE))
        if str(self.cfg.value("quality_v102", "")) != "1":
            if _ms < 720:
                _ms = gifio.MAX_SHORT_SIDE
            self.cfg.setValue("quality_v102", "1")
            self.cfg.setValue("max_short", _ms)
        self.opt = dict(
            enhance=str(self.cfg.value("enhance", "gpen256")),
            color_match=self.cfg.value("color_match", "false") in (True, "true", "1", 1),
            color_ref_path=str(self.cfg.value("color_ref_path", "") or ""),
            seamless=self.cfg.value("seamless", "false") in (True, "true", "1", 1),
            temporal_smooth=float(self.cfg.value("temporal_smooth", 0.12)),
            min_confidence=float(self.cfg.value("min_confidence", 0.55)),
            max_short=_ms,
            export_mp4=self.cfg.value("export_mp4", "false") in (True, "true", "1", 1),
            device=dev,
        )
        if self.opt["enhance"] not in ("off", "gpen256", "gpen512"):
            self.opt["enhance"] = "gpen256"

        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(10)
        v.addWidget(self._topbar())
        self.stack = QStackedWidget()
        v.addWidget(self.stack, 1)
        self.stack.addWidget(self._page_setup())
        self.stack.addWidget(self._page_main())
        self.stack.addWidget(self._page_options())
        self.stack.addWidget(self._page_progress())
        self.stack.addWidget(self._page_done())
        self.setAcceptDrops(True)
        self._ensure_engine_hooks()
        if store.all_installed():
            self.go(PAGE_MAIN)
            QTimer.singleShot(200, self._start_bench)
        else:
            self.go(PAGE_SETUP)

    def _topbar(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.title = label("GIF Face Swap", "title")
        lay.addWidget(self.title)
        lay.addStretch(1)
        self.chip = label("…", "hint")
        lay.addWidget(self.chip)
        self.btn_opts_top = button("Options", None, 140)
        self.btn_opts_top.clicked.connect(lambda: self.go(PAGE_OPTIONS))
        lay.addWidget(self.btn_opts_top)
        about = button("About", None, 120)
        about.clicked.connect(self._about)
        lay.addWidget(about)
        return w

    def set_chip(self):
        try:
            eng = self.job.engine
            if eng and getattr(eng, "info", None):
                self.chip.setText(eng.info.label())
                return
        except Exception:  # noqa: BLE001
            pass
        self.chip.setText(f"device: {self.opt.get('device', 'auto')}")

    def _toast_gpu_fallback(self, reason=""):
        msg = "GPU failed, using CPU"
        if reason:
            msg += f"\n{reason[:160]}"
        QTimer.singleShot(0, lambda: QMessageBox.information(self, "DirectML", msg))

    def _ensure_engine_hooks(self):
        try:
            eng = self.job.get_engine(self.opt.get("device", "auto"))
            eng.on_fallback(self._toast_gpu_fallback)
            if eng.info.fell_back:
                QTimer.singleShot(200, lambda: self._toast_gpu_fallback(eng.info.fallback_reason))
            self.set_chip()
        except Exception as e:  # noqa: BLE001
            log.warning("engine hook: %s", e)

    def _about(self):
        QMessageBox.information(
            self, "About",
            f"GIF Face Swap {__version__}\n"
            "FaceFusion-class models (YOLO + ArcFace + InSwapper + GPEN).\n"
            "Face-only blend by default (optional LAB colour match off).\n"
            "Unsigned build — SmartScreen: More info → Run anyway.\n"
            "Models: InsightFace non-commercial research unless licensed.\n"
            f"Crash log: %LOCALAPPDATA%\\GifFaceSwap\\crash.log",
        )

    def go(self, page):
        self.stack.setCurrentIndex(page)
        self.btn_opts_top.setVisible(page in (PAGE_MAIN, PAGE_DONE))
        if page == PAGE_OPTIONS:
            self._refresh_options()
        if page == PAGE_MAIN:
            self._update_summary()
        if page == PAGE_SETUP:
            self._refresh_setup()

    # ---- setup / models ----
    def _page_setup(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(label("1 · Download AI models", "steplabel"))
        lay.addWidget(label(
            f"First run needs ~{REQUIRED_BYTES / 1e6:.0f} MB (YOLO Face + ArcFace + inswapper). "
            "Optional Light/HQ enhancers later. Reuses Face Fusion Studio models if already on this PC.",
            "subtitle", True))
        self.setup_bar = QProgressBar()
        self.setup_bar.setRange(0, 1000)
        lay.addWidget(self.setup_bar)
        self.setup_status = label("Ready to download.", "hint", True)
        lay.addWidget(self.setup_status)
        row = QHBoxLayout()
        self.btn_dl = button("Download models", "primary")
        self.btn_dl.clicked.connect(self._download)
        row.addWidget(self.btn_dl, 2)
        self.btn_setup_cancel = button("Cancel")
        self.btn_setup_cancel.clicked.connect(self._cancel)
        row.addWidget(self.btn_setup_cancel)
        lay.addLayout(row)
        lay.addStretch(1)
        return w

    def _refresh_setup(self):
        missing = self.store.missing_required()
        if not missing:
            self.setup_status.setText("Models ready.")
            self.setup_bar.setValue(1000)
        else:
            self.setup_status.setText(f"Missing: {', '.join(s.file for s in missing)}")

    def _download(self):
        if self._busy_worker():
            return
        self.btn_dl.setEnabled(False)

        def work(cancelled, emit):
            def prog(f, done, total, bps, verifying):
                if cancelled():
                    raise Cancelled()
                emit(dict(file=f, done=done, total=total, bps=bps, verifying=verifying))
            self.store.ensure(progress=prog, cancelled=cancelled)
            return True

        self.worker = Worker(work, self)
        self.worker.progressed.connect(self._setup_progress)
        self.worker.done.connect(lambda _: (self.btn_dl.setEnabled(True), self.go(PAGE_MAIN),
                                            QTimer.singleShot(100, self._start_bench)))
        self.worker.failed.connect(lambda m, t: (self.btn_dl.setEnabled(True),
                                                 QMessageBox.warning(self, "Download", m)))
        self.worker.start()

    def _setup_progress(self, d):
        total = max(int(d.get("total") or 1), 1)
        done = int(d.get("done") or 0)
        self.setup_bar.setValue(int(1000 * done / total))
        kind = "verify" if d.get("verifying") else "get"
        self.setup_status.setText(f"{kind} {d.get('file')}  {done/1e6:.0f}/{total/1e6:.0f} MB")

    # ---- main ----
    def _page_main(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        scroll.setWidget(host)
        lay = QVBoxLayout(host)
        lay.setSpacing(12)

        lay.addWidget(label("Create a face-swapped GIF", "title"))
        lay.addWidget(label(
            "Pick an animated GIF and a clear face photo. Faces are blended with a soft "
            "seam (face-only). Optional colour match is off by default in Options.",
            "subtitle", True))

        self.slot_gif = ImageSlot("1 · Animated GIF", "Tap to pick a .gif")
        self.slot_gif.clicked.connect(self._pick_gif)
        self.slot_gif.dropped.connect(self.open_path)
        lay.addWidget(self.slot_gif)

        self.slot_face = ImageSlot("2 · Face from photo", "Tap to pick a face photo")
        self.slot_face.clicked.connect(self._pick_photo)
        self.slot_face.dropped.connect(self.open_path)
        lay.addWidget(self.slot_face)

        self.pair_row = QHBoxLayout()
        self.pair_imgs = []
        for _ in range(4):
            im = QLabel()
            im.setFixedSize(96, 96)
            im.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pair_imgs.append(im)
            self.pair_row.addWidget(im)
        self.pair_row.addStretch(1)
        lay.addLayout(self.pair_row)

        row = QHBoxLayout()
        self.btn_flip = button("Flip faces", None, 160)
        self.btn_flip.clicked.connect(self._flip)
        row.addWidget(self.btn_flip)
        self.summary = label("", "hint", True)
        row.addWidget(self.summary, 1)
        lay.addLayout(row)

        self.btn_start = button("Create face swap GIF", "primary")
        self.btn_start.clicked.connect(self._start)
        lay.addWidget(self.btn_start)
        return scroll

    def _busy_worker(self):
        return self.worker is not None and self.worker.isRunning()

    def _pick_gif(self):
        p, _ = QFileDialog.getOpenFileName(self, "Animated GIF", "", "GIF (*.gif)")
        if p:
            self.set_gif(p)

    def _pick_photo(self):
        p, _ = QFileDialog.getOpenFileName(self, "Face photo", "", "Images (*.jpg *.jpeg *.png *.webp *.bmp)")
        if p:
            self.set_photo(p)

    def open_path(self, p):
        ext = Path(p).suffix.lower()
        if ext in GIF_EXT:
            self.set_gif(p)
        elif ext in IMAGE_EXT:
            self.set_photo(p)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            self.open_path(u.toLocalFile())

    def set_gif(self, p):
        if self._busy_worker():
            return
        self.gif_path = p
        self.slot_gif.info.setText("Reading…")

        def work(cancelled, emit):
            info, frames = gifio.decode(p)
            if cancelled():
                raise Cancelled()
            preview = frames[0].bgr if frames else None
            return dict(info=info, preview=preview)

        self.worker = Worker(work, self)

        def done(res):
            self.gif_info = res["info"]
            self.gif_preview = res["preview"]
            self.slot_gif.set_image(self.gif_preview)
            self.slot_gif.info.setText(
                f"{self.gif_info.width}×{self.gif_info.height} · {self.gif_info.frame_count} frames · "
                f"{self.gif_info.duration_ms/1000:.1f}s · {self.gif_info.file_bytes/1e6:.1f} MB")
            self._update_summary()

        def failed(msg, tb):
            self.slot_gif.clear_image("Tap to pick a .gif")
            QMessageBox.warning(self, "GIF", msg)

        self.worker.done.connect(done)
        self.worker.failed.connect(failed)
        self.worker.start()

    def set_photo(self, p):
        if self._busy_worker():
            return
        self.photo_path = p
        self.slot_face.info.setText("Finding faces…")

        def work(cancelled, emit):
            return load_photo(p, store=self.store, device=self.opt.get("device", "auto"))

        self.worker = Worker(work, self)

        def done(ph):
            self.photo = ph
            self.slot_face.set_image(ph.img)
            self.slot_face.info.setText(f"{len(ph.faces)} face(s) found")
            self._redraw_pairs()
            self._update_summary()

        def failed(msg, tb):
            self.photo = None
            self.slot_face.clear_image("Tap to pick a face photo")
            QMessageBox.warning(self, "Photo", msg)

        self.worker.done.connect(done)
        self.worker.failed.connect(failed)
        self.worker.start()

    def _redraw_pairs(self):
        for im in self.pair_imgs:
            im.clear()
        if not self.photo or not self.photo.faces:
            return
        for i, f in enumerate(self.photo.faces[:4]):
            crop = face_crop(self.photo.img, f, 96)
            self.pair_imgs[i].setPixmap(pix(crop, 96, 96))

    def _flip(self):
        self.rotation = (self.rotation + 1) % 4
        self._update_summary()

    def settings(self) -> Settings:
        enh = self.opt.get("enhance")
        if enh in (None, "off"):
            enh = None
        return Settings(
            max_short=int(self.opt.get("max_short", gifio.MAX_SHORT_SIDE)),
            enhance=enh,
            rotation=self.rotation,
            device=str(self.opt.get("device", "auto")),
            out_dir=str(default_out_dir()),
            min_confidence=float(self.opt.get("min_confidence", 0.55)),
            color_match=bool(self.opt.get("color_match", False)),
            color_ref_path=str(self.opt.get("color_ref_path", "") or ""),
            seamless=bool(self.opt.get("seamless", False)),
            temporal_smooth=float(self.opt.get("temporal_smooth", 0.12)),
            export_mp4=bool(self.opt.get("export_mp4", False)),
        )

    def _update_summary(self):
        parts = []
        if self.gif_info:
            parts.append(f"GIF {self.gif_info.frame_count}f")
        if self.photo:
            parts.append(f"{len(self.photo.faces)} src face(s)")
        if self.rotation:
            parts.append(f"flip×{self.rotation}")
        parts.append("colour match ON" if self.opt.get("color_match", False) else "colour match off")
        if self.opt.get("color_ref_path"):
            parts.append("custom colour ref")
        enh = self.opt.get("enhance", "gpen256")
        parts.append(ENHANCE_LABEL.get(None if enh == "off" else enh, enh))
        parts.append(f"short≤{int(self.opt.get('max_short', gifio.MAX_SHORT_SIDE))}")
        if self.opt.get("export_mp4"):
            parts.append("MP4 export")
        self.summary.setText(" · ".join(parts))
        ok = bool(self.gif_path and self.photo and self.photo.faces)
        self.btn_start.setEnabled(ok and not self._busy_worker())

    def _start_bench(self):
        if self._busy_worker():
            return

        def work(cancelled, emit):
            return self.job.benchmark(self.opt.get("device", "auto"))

        self.worker = Worker(work, self)
        self.worker.done.connect(self._bench_done)
        self.worker.failed.connect(lambda m, t: log.warning("bench: %s", m))
        self.worker.start()

    def _bench_done(self, b):
        self.bench_done = True
        self.set_chip()
        rec = self.job.recommended_enhance()
        if rec and self.opt.get("enhance") == "gpen256" and rec != "gpen256":
            pass  # keep user/default Light unless they change Options
        self._update_summary()

    def _start(self):
        if not (self.gif_path and self.photo and self.photo.faces):
            return
        if self._busy_worker():
            return
        st = self.settings()
        self.go(PAGE_PROGRESS)
        self.prog_bar.setValue(0)
        self.prog_status.setText("Starting…")

        def work(cancelled, emit):
            return self.job.run_gif(self.gif_path, self.photo, st, progress=emit, cancel=cancelled)

        self.worker = Worker(work, self)
        self.worker.progressed.connect(self._prog)
        self.worker.done.connect(self._job_done)
        self.worker.failed.connect(self._job_failed)
        self.worker.start()

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancelled = True

    # ---- options ----
    def _page_options(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        scroll.setWidget(host)
        lay = QVBoxLayout(host)
        lay.addWidget(label("Options", "title"))

        lay.addWidget(label("Enhancer", "section"))
        self.enh_row = QHBoxLayout()
        self.enh_btns = {}
        for text, val in (("Off", "off"), ("Light", "gpen256"), ("HQ", "gpen512")):
            b = button(text)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, v=val: self._set_enhance(v))
            self.enh_btns[val] = b
            self.enh_row.addWidget(b)
        lay.addLayout(self.enh_row)
        self.btn_dl_light = button("Download Light enhancer (~76 MB)")
        self.btn_dl_light.clicked.connect(lambda: self._dl_enh("gpen256"))
        lay.addWidget(self.btn_dl_light)
        self.btn_dl_hq = button("Download HQ enhancer (~284 MB)")
        self.btn_dl_hq.clicked.connect(lambda: self._dl_enh("gpen512"))
        lay.addWidget(self.btn_dl_hq)

        self.chk_color = QCheckBox("Optional face colour match (LAB, face-masked only)")
        lay.addWidget(self.chk_color)
        self.chk_seamless = QCheckBox("Seamless blend (slower, better hairline)")
        lay.addWidget(self.chk_seamless)

        lay.addWidget(label("Optional colour-look reference (face ROI only; colour match must be on)", "hint", True))
        row = QHBoxLayout()
        self.btn_color_ref = button("Pick colour reference…")
        self.btn_color_ref.clicked.connect(self._pick_color_ref)
        row.addWidget(self.btn_color_ref)
        self.btn_clear_ref = button("Clear")
        self.btn_clear_ref.clicked.connect(self._clear_color_ref)
        row.addWidget(self.btn_clear_ref)
        lay.addLayout(row)
        self.lbl_color_ref = label("", "hint", True)
        lay.addWidget(self.lbl_color_ref)

        lay.addWidget(label("Max short side while processing", "section"))
        self.s_short = QSlider(Qt.Orientation.Horizontal)
        self.s_short.setRange(240, gifio.MAX_SHORT_OPTION)
        self.s_short.setSingleStep(16)
        lay.addWidget(self.s_short)
        self.lbl_short = label("", "hint")
        lay.addWidget(self.lbl_short)
        self.s_short.valueChanged.connect(lambda v: self.lbl_short.setText(f"{v} px"))

        self.chk_mp4 = QCheckBox("Also export MP4 (sharper, non-GIF alternative)")
        lay.addWidget(self.chk_mp4)
        lay.addWidget(label("Tip: leave short side at 720+ so output stays near the original size.", "hint", True))

        lay.addWidget(label("Device", "section"))
        drow = QHBoxLayout()
        self.dev_btns = {}
        for text, val in (("Auto", "auto"), ("GPU (DirectML)", "dml"), ("CPU", "cpu")):
            b = button(text)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, v=val: self._set_device(v))
            self.dev_btns[val] = b
            drow.addWidget(b)
        lay.addLayout(drow)
        self.btn_retry_gpu = button("Retry GPU probe")
        self.btn_retry_gpu.clicked.connect(self._retry_gpu)
        lay.addWidget(self.btn_retry_gpu)

        back = button("Back", "primary")
        back.clicked.connect(lambda: (self._save_options(), self.go(PAGE_MAIN)))
        lay.addWidget(back)
        lay.addStretch(1)
        return scroll

    def _refresh_options(self):
        enh = self.opt.get("enhance", "gpen256")
        for v, b in self.enh_btns.items():
            b.setChecked(v == enh)
        self.chk_color.setChecked(bool(self.opt.get("color_match", True)))
        self.chk_seamless.setChecked(bool(self.opt.get("seamless", False)))
        self.s_short.setValue(int(self.opt.get("max_short", gifio.MAX_SHORT_SIDE)))
        self.lbl_short.setText(f"{self.s_short.value()} px")
        self.chk_mp4.setChecked(bool(self.opt.get("export_mp4", False)))
        for v, b in self.dev_btns.items():
            b.setChecked(v == self.opt.get("device", "auto"))
        ref = self.opt.get("color_ref_path") or ""
        self.lbl_color_ref.setText(Path(ref).name if ref else "Using GIF frame face tone (if colour match on)")
        self.btn_dl_light.setEnabled(not self.store.is_installed(ENHANCER_LIGHT))
        self.btn_dl_hq.setEnabled(not self.store.is_installed(ENHANCER_HQ))

    def _set_enhance(self, v):
        self.opt["enhance"] = v
        for k, b in self.enh_btns.items():
            b.setChecked(k == v)

    def _set_device(self, v):
        self.opt["device"] = v
        for k, b in self.dev_btns.items():
            b.setChecked(k == v)

    def _pick_color_ref(self):
        p, _ = QFileDialog.getOpenFileName(self, "Colour look reference", "", "Images (*.jpg *.jpeg *.png *.webp)")
        if p:
            self.opt["color_ref_path"] = p
            self.lbl_color_ref.setText(Path(p).name)

    def _clear_color_ref(self):
        self.opt["color_ref_path"] = ""
        self.lbl_color_ref.setText("Using GIF frame face tone (if colour match on)")

    def _save_options(self):
        self.opt["color_match"] = self.chk_color.isChecked()
        self.opt["seamless"] = self.chk_seamless.isChecked()
        self.opt["max_short"] = int(self.s_short.value())
        self.opt["export_mp4"] = self.chk_mp4.isChecked()
        for k, v in self.opt.items():
            self.cfg.setValue(k, v)
        self._update_summary()

    def _dl_enh(self, which):
        spec = ENHANCE_SPECS[which]
        if self._busy_worker():
            return

        def work(cancelled, emit):
            def prog(f, done, total, bps, verifying):
                if cancelled():
                    raise Cancelled()
                emit(dict(file=f, done=done, total=total))
            self.store.ensure([spec], progress=prog, cancelled=cancelled)
            return True

        self.worker = Worker(work, self)
        self.worker.done.connect(lambda _: (QMessageBox.information(self, "Models", f"{spec.file} ready"),
                                            self._refresh_options()))
        self.worker.failed.connect(lambda m, t: QMessageBox.warning(self, "Download", m))
        self.worker.start()

    def _retry_gpu(self):
        from .. import dml_probe
        dml_probe.clear_status()
        os.environ["GFS_DML_REPROBE"] = "1"
        try:
            self.job.engine = None
            self._ensure_engine_hooks()
            self._start_bench()
            QMessageBox.information(self, "GPU", "Re-probing DirectML…")
        finally:
            os.environ.pop("GFS_DML_REPROBE", None)

    # ---- progress / done ----
    def _page_progress(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(label("Working…", "title"))
        self.prog_status = label("", "subtitle", True)
        lay.addWidget(self.prog_status)
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 1000)
        lay.addWidget(self.prog_bar)
        row = QHBoxLayout()
        self.prog_before = QLabel()
        self.prog_before.setMinimumHeight(200)
        self.prog_after = QLabel()
        self.prog_after.setMinimumHeight(200)
        row.addWidget(self.prog_before, 1)
        row.addWidget(self.prog_after, 1)
        lay.addLayout(row)
        self.btn_cancel = button("Cancel", "danger")
        self.btn_cancel.clicked.connect(self._cancel)
        lay.addWidget(self.btn_cancel)
        lay.addStretch(1)
        return w

    def _prog(self, d):
        stage = d.get("stage", "")
        done, total = int(d.get("done") or 0), max(int(d.get("total") or 1), 1)
        self.prog_bar.setValue(int(1000 * done / total))
        detail = d.get("detail") or ""
        eta = d.get("eta")
        eta_s = f" · ETA {eta:.0f}s" if eta else ""
        self.prog_status.setText(f"{stage}: {done}/{total}{eta_s}  {detail}")
        if d.get("before") is not None:
            self.prog_before.setPixmap(pix(d["before"], 400, 220))
        if d.get("thumb") is not None:
            self.prog_after.setPixmap(pix(d["thumb"], 400, 220))

    def _job_done(self, res):
        self.result = res
        self.set_chip()
        self.go(PAGE_DONE)
        self.done_path.setText(res.get("path", ""))
        info = (f"{res.get('frames')} frames · {res.get('W')}×{res.get('H')} · "
                f"{res.get('total_s')}s · {res.get('size', 0)/1e6:.2f} MB · "
                f"colour match={'on' if res.get('color_match') else 'off'}")
        self.done_info.setText(info)
        if res.get("before") is not None:
            self.done_before.setPixmap(pix(res["before"], 420, 240))
        if res.get("after") is not None:
            self.done_after.setPixmap(pix(res["after"], 420, 240))

    def _job_failed(self, msg, tb):
        if msg == "cancelled":
            self.go(PAGE_MAIN)
            return
        QMessageBox.critical(self, "Failed", msg)
        self.go(PAGE_MAIN)

    def _page_done(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(label("Done", "title"))
        self.done_info = label("", "subtitle", True)
        lay.addWidget(self.done_info)
        row = QHBoxLayout()
        self.done_before = QLabel()
        self.done_before.setMinimumHeight(220)
        self.done_after = QLabel()
        self.done_after.setMinimumHeight(220)
        row.addWidget(self.done_before, 1)
        row.addWidget(self.done_after, 1)
        lay.addLayout(row)
        self.done_path = label("", "hint", True)
        lay.addWidget(self.done_path)
        brow = QHBoxLayout()
        openb = button("Open folder", "primary")
        openb.clicked.connect(self._open_out)
        brow.addWidget(openb)
        again = button("Another GIF")
        again.clicked.connect(lambda: self.go(PAGE_MAIN))
        brow.addWidget(again)
        lay.addLayout(brow)
        lay.addStretch(1)
        return w

    def _open_out(self):
        if not self.result:
            return
        p = Path(self.result["path"])
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.parent)))


def make_app(argv=None):
    # High-DPI Ally X friendly
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(argv or ["GifFaceSwap"])
    app.setApplicationName("GIF Face Swap")
    app.setOrganizationName("vanu")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)
    return app


def run_gui(store: ModelStore, device="auto"):
    app = make_app(sys.argv)
    win = MainWindow(store, device)
    win.resize(1280, 720)
    win.show()
    return app.exec()
