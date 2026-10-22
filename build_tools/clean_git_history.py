"""用 git filter-branch 的 index-filter 清除历史中的硬编码 token。

filter-branch 需要一个 shell 脚本，这里生成临时脚本再执行，
避免 PowerShell 的引号转义问题。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(r"F:\vibe coding工程\feiyu-qqbot")
SCRIPT = Path("C:/Users/Public/filter_token.sh")

# 把所有可能的 token 字面量替换成占位符
SCRIPT.write_text(
    """#!/bin/sh
# 清除历史提交里的硬编码 GitHub token
for f in $(git ls-files 2>/dev/null); do
  case "$f" in
    *.py|*.md|*.yml|*.yaml|*.sh|*.txt|*.iss|*.json|*.cfg|*.toml)
      if [ -f "$f" ]; then
        sed -i -E 's/ghp_[A-Za-z0-9]{20,}/GITHUB_TOKEN_PLACEHOLDER/g' "$f" 2>/dev/null || true
      fi
      ;;
  esac
done
exit 0
""",
    encoding="utf-8",
    newline="\n",
)


def main() -> int:
    env = {"GIT_FILTER_BRANCH_SQUELCH_WARNING": "1"}
    r = subprocess.run(
        ["git", "filter-branch", "-f", "--tree-filter",
         f"sh {SCRIPT.as_posix()}", "--", "--all"],
        cwd=str(ROOT), capture_output=True, text=True, env={**env, **dict(__import__("os").environ)},
    )
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-1500:])
        return 1

    # 清理旧引用
    for cmd in (["git", "reflog", "expire", "--expire=now", "--all"],
                ["git", "gc", "--prune=now", "--aggressive"]):
        subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

    # 验证
    r2 = subprocess.run(
        ["git", "log", "--all", "--oneline", "-S", "ghp_PIiwKdYY"],
        cwd=str(ROOT), capture_output=True, text=True)
    print("仍含 token 的提交:", r2.stdout.strip() or "（无）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
