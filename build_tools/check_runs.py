"""查看 GitHub Actions 运行状态。"""
import json
import os
import sys
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
    repos = sys.argv[1:] or ["szkele1145/feiyu-qqbot", "szkele1145/imdeploy"]
    for repo in repos:
        print(f"\n=== {repo} ===")
        try:
            runs = get(f"/repos/{repo}/actions/runs?per_page=5")
        except Exception as exc:  # noqa: BLE001
            print("查询失败:", exc)
            continue
        for r in runs.get("workflow_runs", []):
            status = r["status"]
            concl = r.get("conclusion") or "-"
            mark = {"completed": "✓", "in_progress": "…", "queued": "·"}.get(status, "?")
            print(f"  {mark} {r['name']:22s} {r['head_branch']:10s} {status:12s} {concl}")
            print(f"     {r['html_url']}")
            if status == "in_progress" or status == "queued":
                jobs = get(f"/repos/{repo}/actions/runs/{r['id']}/jobs")
                for j in jobs.get("jobs", []):
                    print(f"       └ {j['name']}: {j['status']} {j.get('conclusion') or '-'}")


if __name__ == "__main__":
    main()
