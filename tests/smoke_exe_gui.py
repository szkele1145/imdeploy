"""验证 imdeploy exe：启动后能弹出 GUI 面板。"""
import ctypes
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "dist" / "imdeploy" / "imdeploy.exe"


def titles() -> list[str]:
    out = []
    EnumWindows = ctypes.windll.user32.EnumWindows
    GetWindowTextW = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLengthW = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible
    CB = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))

    def cb(hwnd, _l):
        if IsWindowVisible(hwnd):
            n = GetWindowTextLengthW(hwnd)
            if n:
                b = ctypes.create_unicode_buffer(n + 1)
                GetWindowTextW(hwnd, b, n + 1)
                out.append(b.value)
        return True

    EnumWindows(CB(cb), None)
    return out


def main() -> int:
    if not EXE.exists():
        print(f"[错误] 找不到 {EXE}")
        return 1
    p = subprocess.Popen([str(EXE)], cwd=str(EXE.parent))
    hit = None
    try:
        for _ in range(20):
            time.sleep(1)
            for t in titles():
                if "模型" in t or "ImDeploy" in t or "imdeploy" in t.lower():
                    hit = t
                    break
            if hit:
                break
    finally:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()

    if hit:
        print(f"OK: 面板已弹出 -> 「{hit}」")
        return 0
    print("FAIL: 20 秒内未检测到窗口")
    print("可见窗口:", titles()[:12])
    return 1


if __name__ == "__main__":
    sys.exit(main())
