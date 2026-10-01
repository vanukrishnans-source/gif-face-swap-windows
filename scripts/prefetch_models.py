"""Prefetch required (+ optional light enhancer) models into GFS_MODELS / .models-cache."""
from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gfs.models import ModelStore, REQUIRED, ENHANCER_LIGHT, ENHANCER_HQ

def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "required").lower()
    specs = list(REQUIRED)
    if mode in ("light", "all"):
        specs.append(ENHANCER_LIGHT)
    if mode in ("hq", "all"):
        specs.append(ENHANCER_HQ)
    store = ModelStore()
    print("models dir:", store.dir)
    def prog(f, done, total, bps, verifying):
        print(f"  {'verify' if verifying else 'get'} {f} {done/1e6:.1f}/{total/1e6:.1f} MB", flush=True)
    store.ensure(specs, progress=prog)
    print("ok")
if __name__ == "__main__":
    main()
