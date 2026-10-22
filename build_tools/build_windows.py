"""Windows 打包：PyInstaller onedir + Inno Setup 安装程序。

用法::

    python build_tools/build_windows.py            # exe + 安装程序
    python build_tools/build_windows.py --exe-only # 只出 exe
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "imdeploy"
DISPLAY_NAME = "ImDeploy 本地模型部署器"
VERSION = "0.1.0"

HIDDEN = [
    "imdeploy.gui",
    "imdeploy.api_server",
    "imdeploy.backends",
    "imdeploy.catalog",
    "imdeploy.recommend",
    "imdeploy.hardware",
    "imdeploy.library",
    "imdeploy.downloader",
    "imdeploy.proxy",
    "yaml",
]


def build_exe() -> Path:
    pyinstaller = shutil.which("pyinstaller")
    if not pyinstaller:
        alt = ROOT / ".venv" / "Scripts" / "pyinstaller.exe"
        if alt.exists():
            pyinstaller = str(alt)
        else:
            raise SystemExit("未找到 pyinstaller，请先 pip install pyinstaller")

    entry = ROOT / "build_tools" / "entry.py"
    cmd = [
        pyinstaller,
        "--noconfirm", "--clean",
        "--onedir",
        "--name", APP_NAME,
        "--paths", str(ROOT / "src"),
        "--workpath", str(ROOT / "build" / "pyinstaller"),
        "--specpath", str(ROOT / "build"),
        "--distpath", str(ROOT / "dist"),
        "--console",
    ]
    for h in HIDDEN:
        cmd += ["--hidden-import", h]
    cmd.append(str(entry))

    print("执行:", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(ROOT))
    if rc.returncode != 0:
        raise SystemExit(f"PyInstaller 失败，退出码 {rc.returncode}")

    exe = ROOT / "dist" / APP_NAME / f"{APP_NAME}.exe"
    if not exe.exists():
        raise SystemExit(f"未生成 {exe}")

    # 附带默认目录
    for sub in ("data", "models", "logs"):
        (exe.parent / sub).mkdir(exist_ok=True)
    return exe


ISS = r"""; ImDeploy 安装脚本（由 build_windows.py 生成）
#define MyAppName "ImDeploy 本地模型部署器"
#define MyAppVersion "@VERSION@"
#define MyAppExeName "imdeploy.exe"

[Setup]
AppId={{8F2A6C31-4D7B-4E2A-9C51-3B7E5D9A0001}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\ImDeploy
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=@OUTDIR@
OutputBaseFilename=imdeploy-setup-@VERSION@
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"

[Files]
Source: "@SRC@\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{app}\models"
Name: "{app}\data"
Name: "{app}\logs"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "立即启动"; Flags: nowait postinstall skipifsilent
"""


def find_iscc() -> Path | None:
    for cand in [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    ]:
        if cand.exists():
            return cand
    found = shutil.which("iscc")
    return Path(found) if found else None


def build_installer(exe: Path) -> Path | None:
    iscc = find_iscc()
    if not iscc:
        print("[提示] 未找到 Inno Setup 6，跳过安装程序")
        return None

    out_dir = ROOT / "dist" / "installer"
    out_dir.mkdir(parents=True, exist_ok=True)

    iss_path = ROOT / "build_tools" / "installer.iss"
    iss_path.write_text(
        ISS.replace("@VERSION@", VERSION)
           .replace("@OUTDIR@", str(out_dir))
           .replace("@SRC@", str(exe.parent)),
        encoding="utf-8-sig",
    )

    rc = subprocess.run([str(iscc), str(iss_path)], cwd=str(ROOT))
    if rc.returncode != 0:
        print(f"[错误] ISCC 失败，退出码 {rc.returncode}")
        return None

    target = out_dir / f"imdeploy-setup-{VERSION}.exe"
    if target.exists():
        return target
    print("[错误] 未找到生成的安装程序")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe-only", action="store_true")
    args = parser.parse_args(argv)

    exe = build_exe()
    print(f"[完成] exe: {exe}  ({exe.stat().st_size / 1024 / 1024:.1f} MB)")

    if args.exe_only:
        return 0

    setup = build_installer(exe)
    if setup:
        print(f"[完成] 安装程序: {setup}  ({setup.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
