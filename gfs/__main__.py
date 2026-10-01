import os, sys
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
try:
    from gfs.crashlog import install as _install_crashlog
    _install_crashlog()
except Exception:
    pass
from gfs.cli import main
raise SystemExit(main())
