"""Linux 发布包：源码 + 一键安装脚本，打成 tar.gz。

形态与 feiyu-qqbot 对齐：一个压缩包，内置 .sh 自动安装。

用法::

    python build_tools/build_linux.py
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "imdeploy-linux-x86_64"

#: 打进压缩包的顶层文件
INCLUDE_FILES = [
    "requirements.txt",
    "README.md",
    "LICENSE",
]

#: 打进压缩包的目录
INCLUDE_DIRS = [
    "src/imdeploy",
]

INSTALL_SH = r"""#!/usr/bin/env bash
# ImDeploy 本地模型部署器 —— Linux / macOS 一键安装脚本
set -euo pipefail

APP_NAME="imdeploy"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/share/$APP_NAME}"
VENV_DIR="$INSTALL_DIR/venv"
USE_VENV=1

for arg in "$@"; do
  case "$arg" in
    --no-venv) USE_VENV=0 ;;
    --help|-h)
      echo "用法: bash install.sh [--no-venv]"
      exit 0 ;;
  esac
done

info() { echo -e "\033[36m[信息]\033[0m $*"; }
ok()   { echo -e "\033[32m[完成]\033[0m $*"; }
warn() { echo -e "\033[33m[警告]\033[0m $*"; }
fail() { echo -e "\033[31m[错误]\033[0m $*" >&2; exit 1; }

detect_pm() {
  if command -v apt-get >/dev/null 2>&1; then echo apt
  elif command -v dnf >/dev/null 2>&1; then echo dnf
  elif command -v yum >/dev/null 2>&1; then echo yum
  elif command -v pacman >/dev/null 2>&1; then echo pacman
  elif command -v apk >/dev/null 2>&1; then echo apk
  else echo none; fi
}

install_deps() {
  local pm; pm=$(detect_pm)
  info "包管理器: $pm"
  case "$pm" in
    apt)  SUDO=sudo; [ "$(id -u)" = "0" ] && SUDO=""
          $SUDO apt-get update -y
          $SUDO apt-get install -y python3 python3-venv python3-pip ;;
    dnf)  SUDO=sudo; [ "$(id -u)" = "0" ] && SUDO=""
          $SUDO dnf install -y python3 python3-pip ;;
    yum)  SUDO=sudo; [ "$(id -u)" = "0" ] && SUDO=""
          $SUDO yum install -y python3 python3-pip ;;
    pacman) SUDO=sudo; [ "$(id -u)" = "0" ] && SUDO=""
          $SUDO pacman -S --noconfirm python python-pip ;;
    apk)  SUDO=su -c
          $SUDO "apk add --no-cache python3 py3-pip" ;;
    *) warn "未识别包管理器，跳过系统依赖安装" ;;
  esac
}

info "安装目录: $INSTALL_DIR"
install_deps
mkdir -p "$INSTALL_DIR"

if [ "$USE_VENV" = "1" ]; then
  command -v python3 >/dev/null 2>&1 || fail "未找到 python3，请先装 Python 3.9+"
  python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' \
    || fail "需要 Python 3.9 及以上"
  info "创建虚拟环境…"
  python3 -m venv "$VENV_DIR" 2>/dev/null || python3 -m venv --without-pip "$VENV_DIR"
  PIP="$VENV_DIR/bin/pip"
  [ -x "$PIP" ] || PIP="python3 -m pip"
  info "安装 Python 依赖…"
  "$PIP" install --upgrade pip -q || true
  "$PIP" install -q -r "$REPO_DIR/requirements.txt" \
    || "$PIP" install -q aiohttp httpx pyyaml || fail "依赖安装失败"
  ok "虚拟环境就绪: $VENV_DIR"
else
  PY=python3
fi

info "复制程序文件…"
mkdir -p "$INSTALL_DIR/src"
cp -r "$REPO_DIR/src/imdeploy" "$INSTALL_DIR/src/"
mkdir -p "$INSTALL_DIR"/{data,models,logs}

cat > "$INSTALL_DIR/run.sh" <<EOF
#!/usr/bin/env bash
cd "$INSTALL_DIR"
export PYTHONPATH="$INSTALL_DIR/src:\$PYTHONPATH"
exec "$VENV_DIR/bin/python" -m imdeploy "\$@"
EOF
chmod +x "$INSTALL_DIR/run.sh"

cat > "$INSTALL_DIR/run-api.sh" <<EOF
#!/usr/bin/env bash
cd "$INSTALL_DIR"
export PYTHONPATH="$INSTALL_DIR/src:\$PYTHONPATH"
exec "$VENV_DIR/bin/python" -m imdeploy --serve "\$@"
EOF
chmod +x "$INSTALL_DIR/run-api.sh"

if [ ! -f "$INSTALL_DIR/data/config.yaml" ]; then
  info "生成配置模板…"
  ( cd "$INSTALL_DIR" && PYTHONPATH="$INSTALL_DIR/src" python3 -m imdeploy --init ) || true
fi

ok "安装完成！"
echo
echo "启动面板:  $INSTALL_DIR/run.sh"
echo "只起 API:  $INSTALL_DIR/run-api.sh"
echo "API 地址:  http://127.0.0.1:8080"
echo "配置文件:  $INSTALL_DIR/data/config.yaml"
"""

START_SH = """#!/usr/bin/env bash
# 启动 ImDeploy（弹 WebUI 面板）
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/share/imdeploy}"
if [ -x "$INSTALL_DIR/run.sh" ]; then
  exec "$INSTALL_DIR/run.sh" "$@"
fi
export PYTHONPATH="$APP_DIR/src:${PYTHONPATH:-}"
exec python3 -m imdeploy "$@"
"""

RUN_API_SH = """#!/usr/bin/env bash
# 只启动 API 服务（无界面）
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/share/imdeploy}"
if [ -x "$INSTALL_DIR/run-api.sh" ]; then
  exec "$INSTALL_DIR/run-api.sh" "$@"
fi
export PYTHONPATH="$APP_DIR/src:${PYTHONPATH:-}"
exec python3 -m imdeploy --serve "$@"
"""


def _add_file(tar: tarfile.TarFile, src: Path, arcname: str) -> None:
    data = src.read_bytes()
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mode = 0o755 if os.access(src, os.X_OK) else 0o644
    info.mtime = int(src.stat().st_mtime)
    tar.addfile(info, io.BytesIO(data))


def _add_exec(tar: tarfile.TarFile, arcname: str, content: str) -> None:
    data = content.encode("utf-8")
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mode = 0o755
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(data))


def _add_dir(tar: tarfile.TarFile, src: Path, prefix: str) -> int:
    count = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        _add_file(tar, path, f"{prefix}/{path.relative_to(src).as_posix()}")
        count += 1
    return count


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{NAME}.tar.gz"

    for rel in INCLUDE_DIRS:
        if not (ROOT / rel).is_dir():
            raise SystemExit(f"[错误] 缺少目录 {rel}")

    with tarfile.open(target, "w:gz") as tar:
        total = 0
        for rel in INCLUDE_FILES:
            src = ROOT / rel
            if not src.exists():
                print(f"[跳过] {rel} 不存在")
                continue
            _add_file(tar, src, f"{NAME}/{rel}")
            total += 1

        for rel in INCLUDE_DIRS:
            total += _add_dir(tar, ROOT / rel, f"{NAME}/{rel}")

        _add_exec(tar, f"{NAME}/install.sh", INSTALL_SH)
        _add_exec(tar, f"{NAME}/start.sh", START_SH)
        _add_exec(tar, f"{NAME}/run-api.sh", RUN_API_SH)
        total += 3

    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"[完成] {target}  ({size_mb:.2f} MB, {total} 个文件)")
    print("       解压后执行: bash install.sh")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="打包 Linux 发行包")
    parser.add_argument("--out", default=str(ROOT / "dist"))
    args = parser.parse_args(argv)
    build(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
