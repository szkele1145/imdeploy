"""把 exe 里的图标资源逐张导出成 PNG，验证像素数据是否真的有效。"""
import struct
import sys
from pathlib import Path

import pefile

exe = sys.argv[1]
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

pe = pefile.PE(exe)
RT_ICON = 3
RT_GROUP_ICON = 14

# 收集 RT_ICON 原始数据
icons = {}
for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries:
    if entry.id != RT_ICON:
        continue
    for item in entry.directory.entries:
        for lang in item.directory.entries:
            data = lang.data.struct
            icons[item.id] = pe.get_data(data.OffsetToData, data.Size)

print(f"RT_ICON 数量: {len(icons)}")

# 读 RT_GROUP_ICON
for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries:
    if entry.id != RT_GROUP_ICON:
        continue
    for grp in entry.directory.entries:
        for lang in grp.directory.entries:
            data = lang.data.struct
            raw = pe.get_data(data.OffsetToData, data.Size)
            reserved, itype, count = struct.unpack("<HHH", raw[:6])
            print(f"\n组 {grp.id}: reserved={reserved} type={itype} count={count}")
            for i in range(count):
                off = 6 + i * 14
                # GRPICONDIRENTRY: bWidth,bHeight,bColorCount,bReserved,wPlanes,
                # wBitCount,dwBytesInRes(4),nID(2) = 14 字节
                (w, h, colors, res, planes, bits,
                 size, nid) = struct.unpack("<BBBBHHIH", raw[off:off + 14])
                real_w = w or 256
                real_h = h or 256
                blob = icons.get(nid)
                status = "OK" if blob else f"!!! 缺少 RT_ICON {nid}"
                if blob:
                    is_png = blob[:4] == b"\x89PNG"
                    head = "PNG" if is_png else "BMP"
                    if len(blob) != size:
                        status += f" 大小不符(声明{size} 实际{len(blob)})"
                    else:
                        status += f" {head} {len(blob)}B"
                    ext = ".png" if is_png else ".bmp"
                    (out_dir / f"icon_{real_w}x{real_h}{ext}").write_bytes(blob)
                print(f"  #{i} {real_w}x{real_h} {bits}bpp id={nid} -> {status}")
