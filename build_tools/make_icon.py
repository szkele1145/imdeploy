"""生成 ImDeploy 应用图标（纯 Python，无需 Pillow）。

画一个深色圆角方块 + 蓝色神经网络节点，输出多尺寸 .ico。
"""

from __future__ import annotations

import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build_tools" / "app.ico"

BG = (13, 17, 23)        # #0d1117
ACCENT = (88, 166, 255)  # #58a6ff
ACCENT2 = (63, 185, 80)  # #3fb950
TEXT = (230, 237, 243)   # #e6edf3


def rounded_rect(px, size, radius, color):
    r = radius
    for y in range(size):
        for x in range(size):
            dx = min(x, size - 1 - x)
            dy = min(y, size - 1 - y)
            if dx >= r or dy >= r:
                px[y][x] = color
            else:
                # 圆角
                cx = r - dx
                cy = r - dy
                if cx * cx + cy * cy <= r * r:
                    px[y][x] = color


def disc(px, size, cx, cy, rad, color):
    for y in range(max(0, int(cy - rad) - 1), min(size, int(cy + rad) + 2)):
        for x in range(max(0, int(cx - rad) - 1), min(size, int(cx + rad) + 2)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= rad * rad:
                px[y][x] = color


def line(px, size, x0, y0, x1, y1, color, w=1):
    steps = max(int(abs(x1 - x0)), int(abs(y1 - y0)), 1) * 2
    for i in range(steps + 1):
        t = i / steps
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        for oy in range(-w, w + 1):
            for ox in range(-w, w + 1):
                xi, yi = int(x) + ox, int(y) + oy
                if 0 <= xi < size and 0 <= yi < size:
                    px[yi][xi] = color


def render(size: int) -> list[list[tuple[int, int, int]]]:
    px = [[BG for _ in range(size)] for _ in range(size)]
    s = size
    rounded_rect(px, s, max(2, s // 8), BG)

    # 三个节点：中心 + 左上 + 右下
    cx = cy = s * 0.5
    n1 = (s * 0.28, s * 0.30)
    n2 = (s * 0.72, s * 0.70)
    n3 = (s * 0.72, s * 0.30)
    w = max(1, s // 32)

    line(px, s, n1[0], n1[1], cx, cy, ACCENT, w)
    line(px, s, cx, cy, n2[0], n2[1], ACCENT, w)
    line(px, s, n1[0], n1[1], n3[0], n3[1], (40, 70, 110), w)
    line(px, s, n3[0], n3[1], n2[0], n2[1], (40, 70, 110), w)

    r = max(2, s * 0.085)
    disc(px, s, n1[0], n1[1], r, ACCENT)
    disc(px, s, n2[0], n2[1], r, ACCENT2)
    disc(px, s, n3[0], n3[1], r * 0.8, (70, 110, 160))
    disc(px, s, cx, cy, r * 1.15, TEXT)
    return px


def to_bmp(px, size):
    """ICO 里的 BMP（32 位 BGRA，自下而上，无文件头）。"""
    header = struct.pack("<IiiHHIIiiII", 40, size, size * 2, 1, 32, 0,
                         size * size * 4, 0, 0, 0, 0)
    rows = []
    for y in range(size - 1, -1, -1):
        row = bytearray()
        for x in range(size):
            r, g, b = px[y][x]
            row += bytes((b, g, r, 255))
        rows.append(bytes(row))
    return header + b"".join(rows)


def render_icon(size: int):
    return render(size)


def write_ico_public(path: Path, sizes=(16, 24, 32, 48, 64, 128, 256)) -> Path:
    from icoutil import write_ico as _w

    return _w(path, render, sizes)


if __name__ == "__main__":
    p = write_ico_public(OUT)
    print(f"已生成图标: {p}  ({p.stat().st_size} 字节)")
