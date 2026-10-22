"""生成 Android 应用图标（mipmap 各密度）。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from icogen import downsample, png_bytes, render_imdeploy  # noqa: E402

DENSITIES = {
    "mipmap-mdpi": 48,
    "mipmap-hdpi": 72,
    "mipmap-xhdpi": 96,
    "mipmap-xxhdpi": 144,
    "mipmap-xxxhdpi": 192,
}


def main() -> int:
    res_dir = ROOT / "android" / "app" / "src" / "main" / "res"
    for folder, size in DENSITIES.items():
        target = res_dir / folder
        target.mkdir(parents=True, exist_ok=True)
        big = render_imdeploy(size * 4)
        small = downsample(big, size)
        (target / "ic_launcher.png").write_bytes(png_bytes(small, size))
        print(f"  {folder}/ic_launcher.png  {size}x{size}")
    print(f"[完成] Android 图标已生成到 {res_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
