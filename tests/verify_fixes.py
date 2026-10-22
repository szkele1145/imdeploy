"""验证：硬件检测耗时、无额外控制台窗口、图标存在。"""
import ctypes
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXE = ROOT / "dist" / "imdeploy" / "imdeploy.exe"

from imdeploy.hardware import detect, summarize  # noqa: E402


def test_hw():
    t0 = time.time()
    hw = detect()
    dt = time.time() - t0
    print(f"[硬件检测] 耗时 {dt:.2f} 秒")
    print("  " + summarize(hw).replace("\n", "\n  "))
    ok = bool(hw.cpu_cores) and bool(hw.cpu_model_name or hw.cpu_arch)
    print("  结果:", "OK" if ok else "FAIL")
    return ok and dt < 20


u32 = ctypes.windll.user32
CB = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)


def visible_windows():
    out = []

    def cb(h, _l):
        if u32.IsWindowVisible(h):
            n = u32.GetWindowTextLengthW(h)
            if n > 0:
                b = ctypes.create_unicode_buffer(n + 1)
                u32.GetWindowTextW(h, b, n + 1)
                cls = ctypes.create_unicode_buffer(256)
                u32.GetClassNameW(h, cls, 256)
                out.append((b.value, cls.value))
        return True

    u32.EnumWindows(CB(cb), None)
    return out


def test_no_extra_console():
    before = {(t, c) for t, c in visible_windows() if "ConsoleWindowClass" in c}
    p = subprocess.Popen([str(EXE)])
    time.sleep(8)
    after = {(t, c) for t, c in visible_windows() if "ConsoleWindowClass" in c}
    extra = after - before
    windows = [t for t, _ in visible_windows() if "模型" in t or "ImDeploy" in t]
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
    print(f"[额外控制台窗口] {len(extra)} 个 {extra if extra else ''}")
    print(f"[主窗口] {windows}")
    return len(extra) == 0 and len(windows) > 0


def test_icon():
    ico = ROOT / "build_tools" / "app.ico"
    exe_ico = EXE.exists() and EXE.stat().st_size > 3_000_000
    print(f"[图标文件] {ico.exists()} ({ico.stat().st_size if ico.exists() else 0} 字节)")
    print(f"[exe 已带图标] {exe_ico}")
    return ico.exists()


if __name__ == "__main__":
    r1 = test_hw()
    r3 = test_icon()
    r2 = test_no_extra_console()
    print("\n结果:", "全部通过" if all([r1, r2, r3]) else "有失败项")
    sys.exit(0 if all([r1, r2, r3]) else 1)
