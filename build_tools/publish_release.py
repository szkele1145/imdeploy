"""CI 用：把构建产物上传到 GitHub Release。

用法::

    python build_tools/publish_release.py --tag v0.1.0 --assets artifacts

所有 GitHub API 写操作都走 urllib + 显式 UTF-8 编码，避免中文变问号。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
UPLOAD = "https://uploads.github.com"
PROXY = os.environ.get("GITHUB_PROXY", "http://127.0.0.1:65532")

RELEASE_NOTE = """## {title}

{intro}

### 安装

{install}

### 说明

{notes}
"""


def _opener():
    handlers = []
    if PROXY:
        handlers.append(urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
    return urllib.request.build_opener(*handlers)


def call(method: str, url: str, data=None, headers=None, want_json=True):
    h = {
        "Authorization": f"token {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "imdeploy-ci",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    with _opener().open(req, timeout=600) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8")) if (want_json and raw) else None


def repo_slug() -> str:
    slug = os.environ.get("GITHUB_REPOSITORY", "").strip()
    return slug or "szkele1145/imdeploy"


def collect_assets(root: Path) -> list[Path]:
    wanted = (".exe", ".tar.gz", ".apk", ".zip")
    return [p for p in sorted(root.rglob("*"))
            if p.is_file() and p.name.lower().endswith(wanted)]


def ensure_release(slug: str, tag: str, body: str) -> dict:
    try:
        return call("GET", f"{API}/repos/{slug}/releases/tags/{tag}")
    except urllib.error.HTTPError:
        payload = {
            "tag_name": tag,
            "name": f"ImDeploy {tag}",
            "body": body,
            "draft": False,
            "prerelease": False,
        }
        return call(
            "POST", f"{API}/repos/{slug}/releases",
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            {"Content-Type": "application/json; charset=utf-8"},
        )


def upload(slug: str, rid: int, path: Path) -> str:
    name = path.name
    size = path.stat().st_size
    with open(path, "rb") as f:
        data = f.read()
    res = call(
        "POST",
        f"{UPLOAD}/repos/{slug}/releases/{rid}/assets?name={name}",
        data,
        {"Content-Type": "application/octet-stream", "Content-Length": str(size)},
    )
    return res["name"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="上传构建产物到 Release")
    parser.add_argument("--tag", required=True, help="Release 标签，如 v0.1.0")
    parser.add_argument("--assets", required=True, help="产物目录")
    parser.add_argument("--title", default="ImDeploy 本地大模型部署器")
    parser.add_argument("--intro", default="按本机硬件推荐并下载 GGUF 模型，提供 OpenAI 兼容 API。")
    parser.add_argument("--notes", default="模型推理需要 llama-server（llama.cpp 构建）。")
    args = parser.parse_args(argv)

    slug = repo_slug()
    root = Path(args.assets)
    files = collect_assets(root)
    if not files:
        print(f"[错误] {root} 下没有可上传的产物")
        return 1

    body = RELEASE_NOTE.format(
        title=args.title,
        intro=args.intro,
        install=(
            "- **Windows**：下载 `imdeploy-setup-*.exe` 双击安装（中文界面）\n"
            "- **Linux**：解压 `imdeploy-linux-x86_64.tar.gz` 后执行 `bash install.sh`\n"
            "- **Android**：安装 APK，内置 Termux 环境"
        ),
        notes=args.notes,
    )
    rel = ensure_release(slug, args.tag, body)
    rid = rel["id"]
    print(f"Release: {rel['html_url']}")

    for f in files:
        print(f"上传 {f.name}（{f.stat().st_size / 1048576:.1f} MB）…")
        try:
            upload(slug, rid, f)
            print(f"  完成 {f.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [失败] {f.name}: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
