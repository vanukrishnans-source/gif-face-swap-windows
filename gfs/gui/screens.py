"""Capture real UI states to PNGs for README / CI."""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from ..models import ModelStore
from .app import PAGE_MAIN, PAGE_OPTIONS, PAGE_PROGRESS, PAGE_SETUP, MainWindow, make_app


def take_screenshots(args) -> int:
    os.environ.setdefault("QT_SCALE_FACTOR", "1.5")
    out = Path(args.screenshots)
    out.mkdir(parents=True, exist_ok=True)
    app = make_app(["GifFaceSwap"])

    def spin(sec=0.05, until=None, timeout=600):
        t0 = time.time()
        while True:
            app.processEvents()
            time.sleep(0.02)
            if until is None and time.time() - t0 >= sec:
                return True
            if until is not None and until():
                return True
            if time.time() - t0 > timeout:
                return False

    def shot(win, name):
        spin(0.2)
        p = out / f"{name}.png"
        win.grab().save(str(p))
        print("wrote", p)

    tmp = Path(tempfile.mkdtemp(prefix="gfs_setup_"))
    w0 = MainWindow(ModelStore(tmp), "cpu")
    w0.resize(1280, 720)
    w0.show()
    w0.go(PAGE_SETUP)
    shot(w0, "01_setup")
    w0.close()
    shutil.rmtree(tmp, ignore_errors=True)

    store = ModelStore(Path(args.models) if args.models else None)
    win = MainWindow(store, getattr(args, "device", "auto"))
    win.resize(1280, 720)
    win.show()
    spin(until=lambda: win.bench_done or True, timeout=30)
    win.go(PAGE_MAIN)
    shot(win, "03_main_empty")
    if getattr(args, "gif", None) and args.photo:
        win.set_gif(args.gif)
        spin(until=lambda: not win.worker or not win.worker.isRunning(), timeout=600)
        win.set_photo(args.photo)
        spin(until=lambda: win.photo is not None and (not win.worker or not win.worker.isRunning()), timeout=600)
        win.go(PAGE_MAIN)
        shot(win, "04_main_ready")
        win.go(PAGE_OPTIONS)
        shot(win, "05_options")
    print("screenshots done")
    return 0
