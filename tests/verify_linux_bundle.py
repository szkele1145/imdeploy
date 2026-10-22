"""校验 imdeploy Linux 发行包结构。"""
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TGZ = ROOT / "dist" / "imdeploy-linux-x86_64.tar.gz"

with tarfile.open(TGZ, "r:gz") as tar:
    members = tar.getmembers()

names = [m.name for m in members]
print(f"总条目: {len(names)}")

required = [
    "imdeploy-linux-x86_64/install.sh",
    "imdeploy-linux-x86_64/start.sh",
    "imdeploy-linux-x86_64/run-api.sh",
    "imdeploy-linux-x86_64/requirements.txt",
    "imdeploy-linux-x86_64/src/imdeploy/__main__.py",
    "imdeploy-linux-x86_64/src/imdeploy/api_server.py",
    "imdeploy-linux-x86_64/src/imdeploy/gui.py",
]

ok = True
for r in required:
    hit = r in names
    print(f"  {'OK ' if hit else 'MISS'} {r}")
    ok = ok and hit

for m in members:
    if m.name.endswith(("install.sh", "start.sh", "run-api.sh")):
        exec_ok = bool(m.mode & 0o111)
        print(f"  {m.name.split('/')[-1]:12s} mode={oct(m.mode)} "
              f"{'OK' if exec_ok else 'NOT EXECUTABLE'}")
        ok = ok and exec_ok

print("\nBUNDLE", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
