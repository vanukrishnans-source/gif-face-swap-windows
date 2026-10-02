"""CLI modes for GifFaceSwap(.exe): GUI / --selftest / --run-gif / --bench / --screenshots / --dml-probe."""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import __version__

log = logging.getLogger("gfs")


def _setup_logging(verbose=True):
    from . import crashlog
    crashlog.install()
    base = crashlog.log_dir()
    base.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(base / "giffaceswap.log", encoding="utf-8")]
    if verbose and sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers, force=True)


def versions():
    import cv2, onnxruntime as ort
    return dict(
        app=__version__, python=sys.version.split()[0], platform=platform.platform(),
        machine=platform.machine(), cpu_count=os.cpu_count(), onnxruntime=ort.__version__,
        providers=ort.get_available_providers(), opencv=cv2.__version__,
        numpy=np.__version__, frozen=bool(getattr(sys, "frozen", False)),
        detector="yoloface_8n", pipeline="gif",
    )


def _progress_printer(prefix="", stages=None):
    last = [0.0]
    seen = stages if stages is not None else []

    def cb(d):
        now = time.time()
        st = d.get("stage")
        if st and (not seen or seen[-1] != st):
            seen.append(st)
        if st in ("done", "encode") or now - last[0] > 2:
            last[0] = now
            eta = d.get("eta")
            log.info("%s%s %s/%s %.2f/s eta %s %s", prefix, st, d.get("done"), d.get("total"),
                     d.get("rate") or 0, f"{eta:.0f}s" if eta else "-", d.get("detail") or "")
    return cb


def _models(store_dir, need, cache=None):
    from .models import ModelStore
    store = ModelStore(Path(store_dir) if store_dir else None)
    if cache:
        got = store.import_from(cache, need)
        if got:
            log.info("imported from cache %s: %s", cache, got)
    missing = [s for s in need if not store.is_installed(s)]
    if missing:
        log.info("downloading %s (%.1f MB)", [s.file for s in missing], sum(s.bytes for s in missing) / 1e6)
        last = [0.0]

        def prog(f, done, total, bps, verifying):
            if time.time() - last[0] > 5:
                last[0] = time.time()
                log.info("  %s %s %.0f/%.0f MB %.1f MB/s", "verify" if verifying else "get",
                         f, done / 1e6, total / 1e6, bps / 1e6)
        store.ensure(missing, progress=prog)
    return store


def dml_smoke(store, frame, dets, assign, photo):
    from . import core, dml_probe
    from .engine import Engine
    report = dict(ok=False)
    try:
        # Prefer YOLO for probe
        mpath = store.path(__import__("gfs.models", fromlist=["YOLOFACE"]).YOLOFACE)
        ok, reason = dml_probe.probe_directml(str(mpath))
        report["probe_ok"] = ok
        report["probe_reason"] = reason
        eng = Engine(store, "dml" if ok else "cpu")
        eng.prepare(None)
        report["device"] = eng.info.label()
        report["active"] = eng.info.active
        report["ok"] = True
    except Exception as e:  # noqa: BLE001
        report["error"] = f"{type(e).__name__}: {e}"
    return report


def _sim_dml_fallback(store):
    from . import dml_probe
    from .engine import Engine
    os.environ["GFS_FORCE_DML_FAIL"] = "1"
    dml_probe.clear_status()
    try:
        eng = Engine(store, "auto")
        eng.prepare(None)
        alive = True
        label = eng.info.label()
        fell = eng.info.fell_back or eng.info.active == "CPU"
        return dict(ok=alive and fell, alive=alive, label=label, fell_back=fell)
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, error=str(e))
    finally:
        os.environ.pop("GFS_FORCE_DML_FAIL", None)


def selftest(args):
    from . import detect, gifio, models as M
    from .job import Job, Settings, load_photo
    report = dict(ok=False, versions=versions(), checks={}, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    out_dir = Path(args.out or tempfile.mkdtemp(prefix="gfs_selftest_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_path = out_dir / "selftest_report.json"
    try:
        log.info("versions %s", json.dumps(report["versions"]))
        enh = None if args.enhance in (None, "off") else args.enhance
        need = list(M.REQUIRED) + ([M.ENHANCER_LIGHT] if enh == "gpen256" else []) + (
            [M.ENHANCER_HQ] if enh == "gpen512" else [])
        store = _models(args.models, need, args.model_cache)
        report["checks"]["models_sha256_ok"] = all(store.is_installed(s) for s in need)

        work = Path(tempfile.mkdtemp(prefix="gfs_st_"))
        gif_src = Path(args.gif)
        if not gif_src.is_file():
            raise FileNotFoundError(f"test GIF missing: {gif_src}")
        gif_path = work / "tést ü.gif"
        shutil.copy(gif_src, gif_path)
        info, frames = gifio.decode(gif_path)
        report["input"] = dict(path=str(gif_path), width=info.width, height=info.height,
                               frames=info.frame_count, duration_ms=info.duration_ms)
        report["checks"]["gif_decoded"] = info.frame_count >= 1 and len(frames) >= 1

        photo_path = work / "fotó.jpg"
        shutil.copy(args.photo, photo_path)
        photo = load_photo(photo_path, store=store, device=args.device)
        report["checks"]["photo_faces"] = len(photo.faces)

        st = Settings(
            max_short=args.max_short, enhance=enh, device=args.device, out_dir=str(out_dir),
            export_mp4=bool(getattr(args, "export_mp4", False)),
            min_confidence=args.min_confidence, color_match=args.color_match,
            color_ref_path=args.color_ref or "",
            temporal_smooth=args.temporal_smooth, seamless=args.seamless,
        )
        job = Job(store, args.device)
        out = out_dir / "selftest_output.gif"
        stages = []
        t_job = time.perf_counter()
        res = job.run_gif(gif_path, photo, st, out_path=out, progress=_progress_printer(stages=stages))
        report["result"] = {k: v for k, v in res.items() if k not in ("before", "after")}
        report["stages"] = stages
        report["job_s"] = round(time.perf_counter() - t_job, 2)

        c = report["checks"]
        c["reached_decode"] = "decode" in stages
        c["reached_detect"] = "detect" in stages
        c["reached_swap"] = "swap" in stages
        c["reached_encode"] = "encode" in stages or "done" in stages
        c["reached_done"] = "done" in stages
        c["gif_written"] = out.is_file() and out.stat().st_size > 32
        try:
            gifio.verify_gif_file(out)
            c["gif_magic_ok"] = True
        except Exception as e:  # noqa: BLE001
            c["gif_magic_ok"] = False
            report["gif_verify_error"] = str(e)
        # Round-trip decode
        info2, frames2 = gifio.decode(out)
        c["gif_roundtrip_frames"] = info2.frame_count >= 1 and len(frames2) >= 1
        # Informational only — default is off; CLI may enable with --color-match
        c["color_match_flag"] = bool(res.get("color_match"))
        report["output"] = dict(frames=info2.frame_count, size=out.stat().st_size,
                                width=info2.width, height=info2.height)

        if args.dml_smoke:
            report["directml"] = dml_smoke(store, frames[0].bgr, [], [], photo)
            report["dml_fallback_sim"] = _sim_dml_fallback(store)
            c["dml_fallback_sim_ok"] = bool(report["dml_fallback_sim"].get("ok"))

        # Colour-match must be OFF by default (face-only seam blend)
        c["color_match_default_off"] = Settings().color_match is False

        # color_match_flag is optional (off by default); photo_faces is a count
        skip = {"photo_faces", "color_match_flag"}
        report["ok"] = all(bool(x) for k, x in c.items() if k not in skip) and c["photo_faces"] >= 1
    except Exception as e:  # noqa: BLE001
        report["error"] = f"{type(e).__name__}: {e}"
        log.exception("selftest failed")
        report["ok"] = False
    rep_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("report %s ok=%s", rep_path, report["ok"])
    return 0 if report["ok"] else 1


def test_dml_fallback(args):
    from . import models as M
    store = _models(args.models, M.REQUIRED, args.model_cache)
    out = Path(args.out or ".")
    out.mkdir(parents=True, exist_ok=True)
    rep = _sim_dml_fallback(store)
    (out / "dml_fallback_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return 0 if rep.get("ok") else 1


def run_gif_cli(args):
    from . import models as M
    from .job import Job, Settings, load_photo
    enh = None if args.enhance in (None, "off") else args.enhance
    need = list(M.REQUIRED) + ([M.ENHANCER_LIGHT] if enh == "gpen256" else []) + (
        [M.ENHANCER_HQ] if enh == "gpen512" else [])
    store = _models(args.models, need, args.model_cache)
    photo = load_photo(args.photo, store=store, device=args.device)
    st = Settings(
        max_short=args.max_short, enhance=enh, device=args.device, out_dir=args.out or "",
        export_mp4=bool(getattr(args, "export_mp4", False)),
        min_confidence=args.min_confidence, color_match=args.color_match,
        color_ref_path=args.color_ref or "", temporal_smooth=args.temporal_smooth,
        seamless=args.seamless,
    )
    job = Job(store, args.device)
    res = job.run_gif(args.gif, photo, st, progress=_progress_printer())
    print(json.dumps({k: v for k, v in res.items() if k not in ("before", "after")}, indent=2, default=str))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="GifFaceSwap", description="GIF Face Swap for Windows (Ally X)")
    p.add_argument("--version", action="store_true")
    p.add_argument("--models", default=None)
    p.add_argument("--model-cache", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "dml", "cpu"])
    p.add_argument("--enhance", default="gpen256", choices=["off", "gpen256", "gpen512"])
    p.add_argument("--max-short", type=int, default=720)
    p.add_argument("--min-confidence", type=float, default=0.55)
    p.add_argument("--color-match", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--color-ref", default="", help="Optional BGR look image for colour match")
    p.add_argument("--seamless", action="store_true")
    p.add_argument("--temporal-smooth", type=float, default=0.12)
    p.add_argument("--export-mp4", action="store_true",
                   help="Also write a sharper MP4 alongside the GIF")
    p.add_argument("--out", default="")
    p.add_argument("--gif", default="")
    p.add_argument("--photo", default="")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--run-gif", action="store_true")
    p.add_argument("--dml-smoke", action="store_true")
    p.add_argument("--test-dml-fallback", action="store_true")
    p.add_argument("--dml-probe", default=None, metavar="MODEL")
    p.add_argument("--screenshots", default=None)
    p.add_argument("--bench", action="store_true")
    args = p.parse_args(argv)

    if args.version:
        print(__version__)
        return 0
    if args.dml_probe:
        from . import dml_probe
        return dml_probe.run_probe_in_this_process(args.dml_probe)

    _setup_logging(verbose=True)

    if args.selftest:
        if not args.gif or not args.photo:
            log.error("--selftest needs --gif and --photo")
            return 2
        return selftest(args)
    if args.test_dml_fallback:
        return test_dml_fallback(args)
    if args.run_gif:
        if not args.gif or not args.photo:
            log.error("--run-gif needs --gif and --photo")
            return 2
        return run_gif_cli(args)
    if args.bench:
        from . import models as M
        from .job import Job
        store = _models(args.models, M.REQUIRED, args.model_cache)
        print(json.dumps(Job(store, args.device).benchmark(), indent=2))
        return 0
    if args.screenshots is not None:
        from .gui.screens import take_screenshots
        return take_screenshots(args)

    # GUI
    from .models import ModelStore, REQUIRED
    store = ModelStore(Path(args.models) if args.models else None)
    if args.gif or args.photo:
        # prefill via env for GUI optional; just launch
        pass
    from .gui.app import run_gui
    return run_gui(store, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
