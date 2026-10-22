"""PyInstaller 入口（Windows onedir 构建用）。

被 PyInstaller 打进去后，``sys._MEIPASS`` 指向 ``_internal`` 目录；
这里把工作目录切到 exe 所在目录，让 data/ models/ logs/ 落在用户可见位置。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap() -> None:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        os.chdir(exe_dir)
        # 源码不在 sys.path 里时补上（onedir 下 _internal 已有打包产物，无需补）
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass and meipass not in sys.path:
            sys.path.insert(0, meipass)


def main() -> int:
    _bootstrap()
    from imdeploy.__main__ import main as real_main

    return real_main()


if __name__ == "__main__":
    raise SystemExit(main())
