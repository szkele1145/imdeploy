"""生成 ICO 的通用工具：小尺寸用 BMP，256 用 PNG（Windows 硬性要求）。

Windows 对 ICO 的规则：
- 16/24/32/48/64/128 可以用 BMP（32 位 BGRA）
- **256x256 必须是 PNG 压缩格式**，用 BMP 会被静默忽略，导致快捷方式显示白块
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def png_bytes(px: list[list[tuple[int, int, int]]], size: int) -> bytes:
    """把像素数组编码成 PNG（RGBA）。"""
    raw = bytearray()
    for y in range(size):
        raw.append(0)  # filter type 0
        for x in range(size):
            r, g, b = px[y][x]
            raw += bytes((r, g, b, 255))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # 8bit RGBA
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b""))


def bmp_bytes(px: list[list[tuple[int, int, int]]], size: int) -> bytes:
    """32 位 BGRA 的 DIB（自下而上，无文件头）。"""
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


def write_ico(path: Path, render, sizes=(16, 24, 32, 48, 64, 128, 256)) -> Path:
    """``render(size)`` 返回像素数组；256 自动用 PNG。"""
    images = []
    for s in sizes:
        px = render(s)
        data = png_bytes(px, s) if s >= 256 else bmp_bytes(px, s)
        images.append((s, data))

    offset = 6 + 16 * len(images)
    entries, blobs = [], []
    for s, data in images:
        entries.append(struct.pack("<BBBBHHII",
                                   0 if s >= 256 else s,
                                   0 if s >= 256 else s,
                                   0, 0, 1, 32, len(data), offset))
        blobs.append(data)
        offset += len(data)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<HHH", 0, 1, len(images)))
        for e in entries:
            f.write(e)
        for b in blobs:
            f.write(b)
    return path
