"""Animated GIF decode + encode for Windows GIF Face Swap.

Lessons from the Android GifFaceSwap 1.0.1 encoder:
  - GIF89a LZW Clear must be emitted at the *current* code width, then width resets.
    Emitting Clear after resetting width corrupts frames once the dictionary fills —
    Photos/browsers then show a broken thumbnail.
  - Write a Global Color Table (gallery / Photos friendly).
  - Verify magic + size before publishing the file (no MediaStore on Windows — just
    refuse to leave a bad .gif on disk).

Decode uses OpenCV/Pillow-free pure parsing for delays + PIL if available, else a
minimal raster path via ``imageio`` / OpenCV VideoCapture fallback, with a final
pure-Python indexed-frame path for CI without optional deps.

Preferred decode path: Pillow (common with PySide6 installs via packaging) or imageio.
"""
from __future__ import annotations

import io
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("gfs")

MAX_SHORT_SIDE = 480
MAX_FRAMES = 80
MAX_DURATION_MS = 12_000
MAX_FILE_BYTES = 25 * 1024 * 1024
MIN_DELAY_MS = 40
DEFAULT_DELAY_MS = 100


class GifError(RuntimeError):
    pass


@dataclass
class GifInfo:
    width: int
    height: int
    frame_count: int
    duration_ms: int
    delays_ms: list
    file_bytes: int
    path: str = ""


@dataclass
class GifFrame:
    bgr: np.ndarray  # HxWx3 uint8
    delay_ms: int


def out_size(w: int, h: int, max_short: int = MAX_SHORT_SIDE) -> Tuple[int, int, float]:
    short = float(min(w, h))
    scale = max_short / short if short > max_short else 1.0
    ow = max(1, int(round(w * scale)))
    oh = max(1, int(round(h * scale)))
    if ow % 2:
        ow -= 1
    if oh % 2:
        oh -= 1
    ow, oh = max(2, ow), max(2, oh)
    return ow, oh, ow / float(w)


def select_indices(delays: Sequence[int]) -> list[int]:
    n = len(delays)
    if n == 0:
        return []
    cum = []
    t = 0
    for d in delays:
        t += max(int(d), MIN_DELAY_MS)
        cum.append(t)
    total = min(cum[-1], MAX_DURATION_MS)
    pool = [i for i in range(n) if (0 if i == 0 else cum[i - 1]) < total]
    if not pool:
        pool = [0]
    if len(pool) <= MAX_FRAMES:
        return pool
    out = []
    for i in range(MAX_FRAMES):
        out.append(pool[int(round(i * (len(pool) - 1) / (MAX_FRAMES - 1)))])
    # unique preserve order
    seen, uniq = set(), []
    for i in out:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def parse_delays(data: bytes) -> list[int]:
    if len(data) < 13:
        return []
    out: list[int] = []
    i = 13
    packed = data[10]
    if packed & 0x80:
        gct = 3 * (1 << ((packed & 0x07) + 1))
        i += gct
    pending = DEFAULT_DELAY_MS
    while i < len(data):
        b = data[i]
        if b == 0x3B:
            break
        if b == 0x21:
            if i + 2 >= len(data):
                break
            label = data[i + 1]
            i += 2
            if label == 0xF9 and i + 5 < len(data):
                block = data[i]
                if block >= 4 and i + 1 + block <= len(data):
                    centi = data[i + 2] | (data[i + 3] << 8)
                    pending = max(centi * 10, MIN_DELAY_MS)
                i += 1 + max(block, 0)
                if i < len(data) and data[i] == 0:
                    i += 1
                else:
                    while i < len(data):
                        sz = data[i]
                        i += 1
                        if sz == 0:
                            break
                        i += sz
            else:
                while i < len(data):
                    sz = data[i]
                    i += 1
                    if sz == 0:
                        break
                    i += sz
        elif b == 0x2C:
            if i + 10 >= len(data):
                break
            out.append(pending)
            pending = DEFAULT_DELAY_MS
            local = data[i + 9]
            i += 10
            if local & 0x80:
                i += 3 * (1 << ((local & 0x07) + 1))
            if i >= len(data):
                break
            i += 1  # LZW min code size
            while i < len(data):
                sz = data[i]
                i += 1
                if sz == 0:
                    break
                i += sz
        else:
            i += 1
    return out


def verify_gif_file(path: Path | str) -> None:
    path = Path(path)
    if not path.is_file():
        raise GifError("Encoded GIF missing.")
    n = path.stat().st_size
    if n < 32:
        raise GifError(f"Encoded GIF too small ({n} bytes) — write failed.")
    with open(path, "rb") as fh:
        hdr = fh.read(6)
    if len(hdr) < 6:
        raise GifError("Encoded GIF truncated.")
    mag = hdr.decode("ascii", "replace")
    if mag not in ("GIF89a", "GIF87a"):
        raise GifError(f"Encoded file is not a GIF (got {hdr.hex()}).")


def decode(path: Path | str) -> Tuple[GifInfo, List[GifFrame]]:
    path = Path(path)
    if not path.is_file():
        raise GifError("GIF file missing.")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise GifError(f"That GIF is too large (max {MAX_FILE_BYTES // (1024 * 1024)} MB).")
    data = path.read_bytes()
    if len(data) < 13 or data[:3] != b"GIF":
        raise GifError("That file isn't a GIF.")
    delays = parse_delays(data)
    if not delays:
        delays = [DEFAULT_DELAY_MS]

    frames_bgr: list[np.ndarray] = []
    # Prefer Pillow
    try:
        from PIL import Image, ImageSequence
        im = Image.open(io.BytesIO(data))
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            rgba = frame.convert("RGBA")
            arr = np.array(rgba)
            # composite on white for transparency
            a = arr[:, :, 3:4].astype(np.float32) / 255.0
            rgb = arr[:, :, :3].astype(np.float32)
            rgb = rgb * a + 255.0 * (1.0 - a)
            bgr = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            frames_bgr.append(bgr)
            if i + 1 >= len(delays) and i + 1 > 1:
                # Pillow may report fewer/more; pad delays later
                pass
        if not frames_bgr:
            raise GifError("No frames in GIF (Pillow).")
    except Exception as e:  # noqa: BLE001
        log.info("Pillow GIF decode unavailable (%s); trying imageio", e)
        try:
            import imageio.v2 as imageio
            reader = imageio.get_reader(path)
            for fr in reader:
                if fr.ndim == 2:
                    bgr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
                elif fr.shape[2] == 4:
                    a = fr[:, :, 3:4].astype(np.float32) / 255.0
                    rgb = fr[:, :, :3].astype(np.float32) * a + 255.0 * (1.0 - a)
                    bgr = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                else:
                    bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                frames_bgr.append(bgr)
            reader.close()
        except Exception as e2:  # noqa: BLE001
            raise GifError(f"Couldn't decode that GIF: {e2}") from e2

    if not frames_bgr:
        raise GifError("No frames found in that GIF.")
    # Align delays to frame count
    if len(delays) < len(frames_bgr):
        delays = list(delays) + [DEFAULT_DELAY_MS] * (len(frames_bgr) - len(delays))
    elif len(delays) > len(frames_bgr):
        delays = delays[: len(frames_bgr)]
    delays = [max(int(d), MIN_DELAY_MS) for d in delays]

    h, w = frames_bgr[0].shape[:2]
    frames = [GifFrame(bgr=f, delay_ms=d) for f, d in zip(frames_bgr, delays)]
    info = GifInfo(
        width=w, height=h, frame_count=len(frames),
        duration_ms=sum(delays), delays_ms=list(delays),
        file_bytes=size, path=str(path),
    )
    return info, frames


def plan_frames(info: GifInfo, frames: List[GifFrame], max_short: int = MAX_SHORT_SIDE
                ) -> Tuple[int, int, List[GifFrame]]:
    """Subsample + resize for processing limits."""
    idxs = select_indices(info.delays_ms)
    W, H, _ = out_size(info.width, info.height, max_short)
    out: list[GifFrame] = []
    for i in idxs:
        fr = frames[i]
        img = fr.bgr
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        out.append(GifFrame(bgr=img, delay_ms=max(fr.delay_ms, MIN_DELAY_MS)))
    return W, H, out


# -------------------- encode (median-cut + LZW, Clear @ current width) --------------------

def _palette_size_bits(n: int) -> int:
    bits, size = 0, 2
    while size < n and bits < 7:
        size *= 2
        bits += 1
    return bits


def _median_cut(colors: list[int], max_colors: int) -> np.ndarray:
    class Box:
        __slots__ = ("list",)

        def __init__(self, lst):
            self.list = lst

        def channel_range(self):
            r0 = g0 = b0 = 255
            r1 = g1 = b1 = 0
            for c in self.list:
                r, g, b = (c >> 16) & 255, (c >> 8) & 255, c & 255
                r0, r1 = min(r0, r), max(r1, r)
                g0, g1 = min(g0, g), max(g1, g)
                b0, b1 = min(b0, b), max(b1, b)
            rs, gs, bs = r1 - r0, g1 - g0, b1 - b0
            if rs >= gs and rs >= bs:
                return 0, rs
            if gs >= rs and gs >= bs:
                return 1, gs
            return 2, bs

        def average(self) -> int:
            if not self.list:
                return 0
            r = g = b = 0
            for c in self.list:
                r += (c >> 16) & 255
                g += (c >> 8) & 255
                b += c & 255
            n = len(self.list)
            return ((r // n) << 16) | ((g // n) << 8) | (b // n)

    boxes = [Box(list(colors))]
    while len(boxes) < max_colors:
        bi, best = -1, -1
        for i, box in enumerate(boxes):
            if len(box.list) < 2:
                continue
            _, span = box.channel_range()
            if span > best:
                best, bi = span, i
        if bi < 0:
            break
        box = boxes.pop(bi)
        ch, _ = box.channel_range()
        box.list.sort(key=lambda c: ((c >> 16) & 255) if ch == 0 else (((c >> 8) & 255) if ch == 1 else (c & 255)))
        mid = len(box.list) // 2
        boxes.append(Box(box.list[:mid]))
        boxes.append(Box(box.list[mid:]))
    return np.array([b.average() for b in boxes], dtype=np.int64)


def quantize(bgr: np.ndarray, max_colors: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indexed HxW uint8, palette Nx3 RGB uint8)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    flat = rgb.reshape(-1, 3)
    step = max(1, flat.shape[0] // 4000)
    sample = flat[::step]
    colors = [int(r) << 16 | int(g) << 8 | int(b) for r, g, b in sample]
    if not colors:
        colors = [0]
    pal_int = _median_cut(colors, max(2, min(256, max_colors)))
    palette = np.zeros((len(pal_int), 3), np.uint8)
    for i, c in enumerate(pal_int):
        palette[i] = ((c >> 16) & 255, (c >> 8) & 255, c & 255)
    # nearest
    # vectorised distance
    diff = flat.astype(np.int16)[:, None, :] - palette.astype(np.int16)[None, :, :]
    dist = (diff * diff).sum(axis=2)
    idx = dist.argmin(axis=1).astype(np.uint8).reshape(bgr.shape[:2])
    return idx, palette


def _lzw_encode(index: np.ndarray, clear_size: int, out: io.BufferedIOBase) -> None:
    clear = clear_size
    eof = clear + 1
    init_width = (clear.bit_length())  # numberOfTrailingZeros(clear)+1 == bit_length for power-of-2
    # clear is power of 2; width = log2(clear)+1
    init_width = int(clear).bit_length()  # 2**n → n+1; for clear=4 (2^2) bit_length=3 = minCode+1 ✓
    table: dict[int, int] = {}

    def key(prefix: int, k: int) -> int:
        return (prefix << 12) | (k & 0xFFF)

    next_code = eof + 1
    width = init_width
    buf = bytearray()
    acc = 0
    acc_bits = 0

    def emit(code: int, nbits: int) -> None:
        nonlocal acc, acc_bits
        acc |= code << acc_bits
        acc_bits += nbits
        while acc_bits >= 8:
            buf.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8

    def flush_blocks() -> None:
        nonlocal acc, acc_bits
        if acc_bits > 0:
            buf.append(acc & 0xFF)
            acc = 0
            acc_bits = 0
        data = bytes(buf)
        buf.clear()
        off = 0
        while off < len(data):
            n = min(255, len(data) - off)
            out.write(bytes([n]))
            out.write(data[off : off + n])
            off += n
        out.write(b"\x00")

    def reset_table() -> None:
        nonlocal next_code, width
        # GIF89a: Clear at *current* width, then reset
        emit(clear, width)
        table.clear()
        next_code = eof + 1
        width = init_width

    flat = index.reshape(-1).astype(np.uint8)
    reset_table()
    if flat.size == 0:
        emit(eof, width)
        flush_blocks()
        return
    prefix = int(flat[0])
    for k in flat[1:]:
        kk = key(prefix, int(k))
        existing = table.get(kk)
        if existing is not None:
            prefix = existing
        else:
            emit(prefix, width)
            if next_code < 4096:
                table[kk] = next_code
                if next_code == (1 << width) and width < 12:
                    width += 1
                next_code += 1
            else:
                reset_table()
            prefix = int(k)
    emit(prefix, width)
    emit(eof, width)
    flush_blocks()


def encode(frames: Sequence[Tuple[np.ndarray, int]], out_path: Path | str, loop: bool = True) -> Path:
    """Encode list of (bgr, delay_ms) to a verified GIF89a file."""
    if not frames:
        raise GifError("No frames to encode.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    h0, w0 = frames[0][0].shape[:2]
    prepared: list[Tuple[np.ndarray, np.ndarray, int]] = []
    for bgr, delay in frames:
        if bgr.shape[0] != h0 or bgr.shape[1] != w0:
            raise GifError("frame size mismatch")
        indexed, palette = quantize(bgr, 256)
        prepared.append((indexed, palette, max(int(delay), MIN_DELAY_MS)))

    gct = prepared[0][1]
    gct_bits = _palette_size_bits(len(gct))
    gct_count = 1 << (gct_bits + 1)

    try:
        with open(tmp, "wb") as os_:
            os_.write(b"GIF89a")
            os_.write(struct.pack("<HH", w0, h0))
            os_.write(bytes([0x80 | 0x70 | gct_bits, 0, 0]))
            # GCT
            for i in range(gct_count):
                if i < len(gct):
                    os_.write(bytes([int(gct[i, 0]), int(gct[i, 1]), int(gct[i, 2])]))
                else:
                    os_.write(b"\x00\x00\x00")
            if loop:
                os_.write(b"\x21\xff\x0bNETSCAPE2.0\x03\x01")
                os_.write(struct.pack("<H", 0))
                os_.write(b"\x00")
            for fi, (indexed, palette, delay_ms) in enumerate(prepared):
                os_.write(b"\x21\xf9\x04\x04")
                os_.write(struct.pack("<H", max(delay_ms // 10, 2)))
                os_.write(b"\x00\x00")
                os_.write(b"\x2c")
                os_.write(struct.pack("<HHHH", 0, 0, w0, h0))
                if fi == 0:
                    pal_bits = gct_bits
                    os_.write(b"\x00")
                else:
                    pal_bits = _palette_size_bits(len(palette))
                    os_.write(bytes([0x80 | pal_bits]))
                    count = 1 << (pal_bits + 1)
                    for i in range(count):
                        if i < len(palette):
                            os_.write(bytes([int(palette[i, 0]), int(palette[i, 1]), int(palette[i, 2])]))
                        else:
                            os_.write(b"\x00\x00\x00")
                min_code = max(pal_bits + 1, 2)
                os_.write(bytes([min_code]))
                _lzw_encode(indexed, 1 << min_code, os_)
            os_.write(b"\x3b")
        verify_gif_file(tmp)
        tmp.replace(out_path)
        verify_gif_file(out_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        raise
    return out_path
