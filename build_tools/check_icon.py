"""检查 exe 的图标资源，找出为什么 Windows 显示白图标。"""
import sys

import pefile

exe = sys.argv[1] if len(sys.argv) > 1 else r"F:\vibe coding工程\imdeploy\dist\imdeploy\imdeploy.exe"
pe = pefile.PE(exe)

RT_ICON = 3
RT_GROUP_ICON = 14
names = {RT_ICON: "RT_ICON", RT_GROUP_ICON: "RT_GROUP_ICON"}

print(f"文件: {exe}")
print(f"资源目录存在: {hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE')}")
if not hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
    print("!!! 无资源目录")
    raise SystemExit(1)

for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries:
    type_id = entry.id if entry.id is not None else entry.name
    label = names.get(type_id, str(type_id))
    count = len(entry.directory.entries)
    print(f"  资源类型 {label}: {count} 项")

    if type_id == RT_GROUP_ICON:
        for grp in entry.directory.entries:
            for lang in grp.directory.entries:
                data = lang.data.struct
                print(f"    组 {grp.id}: 大小 {data.Size} 偏移 0x{data.OffsetToData:X}")
                raw = pe.get_data(data.OffsetToData, data.Size)
                # ICONDIR header: reserved(2) type(2) count(2)
                import struct
                reserved, itype, icount = struct.unpack("<HHH", raw[:6])
                print(f"      ICONDIR: reserved={reserved} type={itype} count={icount}")
                if reserved != 0 or itype != 1:
                    print("      !!! ICONDIR 头非法 —— Windows 会拒绝整个图标")
                for i in range(icount):
                    off = 6 + i * 14
                    w, h, colors, _, planes, bits, size, image_off = struct.unpack(
                        "<BBBBHHII", raw[off:off + 14])
                    print(f"      #{i}: {w or 256}x{h or 256} {bits}bpp size={size}")
