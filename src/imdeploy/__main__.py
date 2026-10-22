"""imdeploy 命令行入口。

用法::

    imdeploy                # 无参数 -> 桌面 GUI
    imdeploy --cli          # 命令行模式
    imdeploy --serve        # 只起 API 服务，不开 GUI
    imdeploy --init         # 生成配置模板
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from . import __version__
from .config import DEFAULT_CONFIG


def _init_config(path: Path) -> int:
    if path.exists():
        print(f"配置已存在，未覆盖: {path}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(DEFAULT_CONFIG, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"已生成配置模板: {path}")
    return 0


def _serve(cfg) -> int:
    from .api_server import build_server

    httpd, api = build_server(cfg)
    try:
        print(f"API 服务已启动: http://{api.host}:{api.port}")
        print("OpenAI 兼容端点: /v1/chat/completions")
        print("多模型别名端点: /api/qwen, /api/llama, /api/gemma ...")
        print("Ctrl+C 退出")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在退出…")
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:]) if argv is None else list(argv)
    parser = argparse.ArgumentParser(prog="imdeploy",
                                     description="本地大模型部署器")
    parser.add_argument("--config", "-c", default=None, help="配置文件路径")
    parser.add_argument("--init", action="store_true", help="生成配置模板后退出")
    parser.add_argument("--serve", action="store_true", help="只启动 API 服务（无 GUI）")
    parser.add_argument("--cli", action="store_true", help="命令行模式（不弹 GUI）")
    parser.add_argument("--version", action="store_true", help="显示版本")
    args, unknown = parser.parse_known_args(raw)

    if args.version:
        print(f"imdeploy v{__version__}")
        return 0

    config_path = Path(args.config) if args.config else Path("data/config.yaml")

    if args.init:
        return _init_config(config_path)

    from .config import Config

    cfg = Config.load(config_path)

    if args.serve:
        return _serve(cfg)

    if not args.cli:
        # 无参数 / GUI 模式
        try:
            from .gui import run_gui
        except Exception as exc:  # noqa: BLE001
            print(f"图形面板不可用（{exc}），回退到命令行模式")
            return _serve(cfg)
        return run_gui(config_path)

    # CLI 模式：打印帮助
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
