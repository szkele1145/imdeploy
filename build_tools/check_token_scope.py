"""检查 GitHub token 的权限范围（能否提交 workflow 文件）。"""
import json
import os
import urllib.request

TOK = os.environ["GITHUB_TOKEN"]
op = urllib.request.build_opener(
    urllib.request.ProxyHandler({"https": "http://127.0.0.1:65532"}))
H = {"Authorization": f"token {TOK}", "Accept": "application/vnd.github+json",
     "User-Agent": "dsh"}


def get(path):
    req = urllib.request.Request(f"https://api.github.com{path}", headers=H)
    with op.open(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    hdr = get("/rate_limit")
    print("速率限制:", hdr.get("rate", {}))
    try:
        res = get("/user")
        print("登录用户:", res.get("login"), res.get("id"))
    except Exception as exc:  # noqa: BLE001
        print("用户信息失败:", exc)

    # 用一个空提交试探 workflow 权限
    import subprocess

    print("\n尝试提交 workflow 文件（探测 scope）…")
    r = subprocess.run(
        ["git", "check-ignore", "-q", ".github/workflows/probe.yml"],
        cwd=r"F:\vibe coding工程\feiyu-qqbot",
        capture_output=True, text=True)
    print(".github 是否被 gitignore:", "是" if r.returncode == 0 else "否")


if __name__ == "__main__":
    main()
