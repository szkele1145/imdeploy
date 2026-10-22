"""验证安装程序界面是中文：启动安装向导，读窗口标题与控件文本。"""
import ctypes
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "dist" / "installer" / "imdeploy-setup-0.1.0.exe"

u32 = ctypes.windll.user32
EnumWindows = u32.EnumWindows
GetWindowTextW = u32.GetWindowTextW
GetWindowTextLengthW = u32.GetWindowTextLengthW
IsWindowVisible = u32.IsWindowVisible
GetClassNameW = u32.GetClassNameW

CB = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))


def text_of(hwnd):
    n = GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    b = ctypes.create_unicode_buffer(n + 1)
    GetWindowTextW(hwnd, b, n + 1)
    return b.value


def all_texts():
    """枚举所有可见窗口及其子控件文本。"""
    found = []

    def walk(hwnd, depth=0):
        if depth > 3:
            return
        if IsWindowVisible(hwnd):
            t = text_of(hwnd)
            if t:
                found.append(t)
        child = []
        WNDENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        u32.EnumChildWindows(hwnd, WNDENUMPROC(
            lambda h, _l: (walk(h, depth + 1), True)[1]), None)

    EnumWindows(CB(lambda h, _l: (walk(h), True)[1]), None)
    return found


def main():
    if not SETUP.exists():
        print(f"[错误] 找不到 {SETUP}")
        return 1

    p = subprocess.Popen([str(SETUP), "/LANG=chs"])
    texts = []
    try:
        for _ in range(15):
            time.sleep(1)
            texts = all_texts()
            if any("安装" in t or "ImDeploy" in t for t in texts):
                break
    finally:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()

    cn = [t for t in texts if any("\u4e00" <= c <= "\u9fff" for c in t)]
    print(f"窗口/控件文本共 {len(texts)} 条，其中含中文 {len(cn)} 条")
    for t in cn[:12]:
        print("   ", t)

    if cn:
        print("\nRESULT: 安装程序界面为中文")
        return 0
    print("\nRESULT: 未检测到中文文本")
    for t in texts[:12]:
        print("   ", t)
    return 1


if __name__ == "__main__":
    sys.exit(main())
