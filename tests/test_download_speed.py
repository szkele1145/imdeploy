"""验证下载进度里的速度/剩余时间字段。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imdeploy.api_server import ApiServer  # noqa: E402
from imdeploy.config import Config  # noqa: E402


class FakeTask:
    def __init__(self, downloaded, size):
        self.downloaded = downloaded
        self.size = size

    @property
    def percent(self):
        return self.downloaded * 100.0 / self.size if self.size else 0.0


def main() -> int:
    results = []

    def check(name, cond, detail=""):
        results.append((name, cond))
        print(f"  {'OK  ' if cond else 'FAIL'} {name} {detail}")

    cfg = Config.default()
    api = ApiServer(cfg, backends=[])

    # 复刻 start_download 里 progress 的算法，直接验证状态字段
    state = {"downloaded": 0, "total": 0, "percent": 0.0}
    lock_ok = True

    def progress(task, now):
        prev_t = state.get("_t") or 0.0
        prev_b = state.get("_b") or 0
        if prev_t and now - prev_t >= 0.4 and task.downloaded >= prev_b:
            inst = (task.downloaded - prev_b) / (now - prev_t)
            old = state.get("speed_bps") or 0.0
            state["speed_bps"] = inst if not old else old * 0.6 + inst * 0.4
            state["_t"] = now
            state["_b"] = task.downloaded
        elif not prev_t:
            state["_t"] = now
            state["_b"] = task.downloaded
        state["downloaded"] = task.downloaded
        state["total"] = task.size
        state["percent"] = round(task.percent, 1)
        speed = state.get("speed_bps") or 0.0
        remain = task.size - task.downloaded
        state["eta"] = int(remain / speed) if (speed > 1024 and remain > 0) else 0

    total = 2 * 1024 * 1024 * 100  # 200 MB
    t = time.time()
    progress(FakeTask(0, total), t)
    check("首次采样无速度", not state.get("speed_bps"), f"-> {state.get('speed_bps')}")

    # 模拟 1 秒下了 10MB
    progress(FakeTask(10 * 1048576, total), t + 1.0)
    speed = state.get("speed_bps") or 0
    check("算出速度", speed > 0, f"-> {speed/1048576:.2f} MB/s")
    check("速度接近 10MB/s", 8 < speed / 1048576 < 12, f"-> {speed/1048576:.2f}")

    eta = state.get("eta") or 0
    check("算出剩余时间", eta > 0, f"-> {eta}s")
    check("ETA 合理（约 19 秒）", 15 <= eta <= 25, f"-> {eta}s")

    check("下划线字段为内部字段", "_t" in state and "_b" in state)

    # download_status 要过滤下划线字段
    api._downloads["m1"] = dict(state)
    api._downloads["m1"]["status"] = "downloading"
    out = api.download_status()
    keys = set(out["m1"].keys())
    check("对外不暴露 _t/_b", not any(k.startswith("_") for k in keys), f"-> {sorted(keys)}")
    check("保留 speed_bps", "speed_bps" in keys)
    check("保留 eta", "eta" in keys)

    ok = sum(1 for _, c in results if c)
    print(f"\n结果: {ok}/{len(results)} 通过")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
