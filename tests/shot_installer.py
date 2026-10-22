"""启动安装程序并截图，用 OCR 判断界面语言。"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "dist" / "installer" / "imdeploy-setup-0.1.0.exe"
SHOT = ROOT / "build" / "installer_shot.png"


def grab(path: Path) -> bool:
    """用 PowerShell 截全屏。"""
    ps = f"""
Add-Type -AssemblyName System.Windows.Forms,System.Drawing
$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
$bmp.Save('{path}', [System.Drawing.Imaging.ImageFormat]::Png)
"""
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True)
    return path.exists() and r.returncode == 0


def main() -> int:
    if not SETUP.exists():
        print(f"[错误] 找不到 {SETUP}")
        return 1
    SHOT.parent.mkdir(parents=True, exist_ok=True)

    p = subprocess.Popen([str(SETUP), "/LANG=chs"])
    time.sleep(5)
    ok = grab(SHOT)
    time.sleep(1)
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()

    if ok:
        print(f"截图: {SHOT}  ({SHOT.stat().st_size // 1024} KB)")
        return 0
    print("截图失败")
    return 1


if __name__ == "__main__":
    sys.exit(main())
