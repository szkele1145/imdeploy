"""硬件探测：识别 CPU / 内存 / GPU，用于模型体积与上下文长度匹配。

全部使用标准库（ctypes / subprocess / platform），跨 Windows / Linux / macOS / Android(Termux)。
任何探测失败都只影响对应字段，不会让整个探测抛异常。
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

__all__ = ["HardwareInfo", "detect", "summarize", "available_ram_gb"]

logger = logging.getLogger(__name__)

_GB = 1024.0**3


@dataclass
class HardwareInfo:
    """本机硬件概况。"""

    cpu_arch: str = ""
    cpu_cores: int = 0
    cpu_model_name: str = ""
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_backend: str = "cpu"
    is_arm: bool = False
    is_android: bool = False
    #: 推理精度偏好（fp16 / q4 / int8 等），由 GPU 能力推断；None 表示未指定
    quant: str | None = "q4"
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def has_gpu(self) -> bool:
        return self.gpu_backend not in ("", "cpu") and self.gpu_vram_gb > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_arch": self.cpu_arch,
            "cpu_cores": self.cpu_cores,
            "cpu_model_name": self.cpu_model_name,
            "total_ram_gb": round(self.total_ram_gb, 2),
            "available_ram_gb": round(self.available_ram_gb, 2),
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": round(self.gpu_vram_gb, 2),
            "gpu_backend": self.gpu_backend,
            "is_arm": self.is_arm,
            "is_android": self.is_android,
            "quant": self.quant,
        }

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return summarize(self)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], timeout: float = 6.0) -> str:
    """执行命令并返回 stdout；失败返回空串。

    关键是 ``CREATE_NO_WINDOW``：否则每调一次 powershell / nvidia-smi
    都会闪一个黑窗口出来，用户会看到一堆莫名其妙的控制台。
    """
    kwargs: dict[str, Any] = {}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = startupinfo
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("命令执行失败 %s: %s", cmd, exc)
        return ""
    return completed.stdout or ""


def _is_android() -> bool:
    if "ANDROID_ROOT" in os.environ or "ANDROID_DATA" in os.environ:
        return True
    if "ANDROID_BOOTLOGO" in os.environ:
        return True
    if platform.system().lower() == "linux":
        try:
            with open("/system/build.prop", "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            if "ro.build.version.release" in text or "ro.product.model" in text:
                return True
        except OSError:
            pass
    return False


def _is_arm(arch: str) -> bool:
    low = (arch or "").lower()
    return low in ("arm64", "aarch64", "armv8l", "armv8b", "arm", "armv7l", "armv6l")


# --------------------------------------------------------------------------- #
# 内存
# --------------------------------------------------------------------------- #
def _memory_windows() -> tuple[float, float]:
    """Windows：用 ctypes 调 GlobalMemoryStatusEx。"""
    total = 0.0
    avail = 0.0

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        stat = MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(MemoryStatusEx)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        if ok:
            total = stat.ullTotalPhys / _GB
            avail = stat.ullAvailPhys / _GB
    except Exception as exc:  # noqa: BLE001
        logger.debug("Windows 内存探测失败: %s", exc)

    if total <= 0:
        # 兜底：用 .NET 性能计数器
        out = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "$c=Get-CimInstance Win32_OperatingSystem;"
                "$c.TotalVisibleMemorySize/1MB;$c.FreePhysicalMemory/1MB",
            ]
        )
        numbers = [float(x) for x in re.findall(r"[\d.]+", out)]
        if len(numbers) >= 2:
            total, avail = numbers[0], numbers[1]
    return total, avail


def _memory_linux() -> tuple[float, float]:
    """Linux / Android(Termux)：读 /proc/meminfo。"""
    total = 0.0
    avail = 0.0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except OSError:
        return total, avail
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        value = rest.strip().split()
        if not value:
            continue
        try:
            kb = int(value[0])
        except ValueError:
            continue
        if key == "MemTotal":
            total = kb / 1024 / 1024
        elif key in ("MemAvailable", "MemFree"):
            if key == "MemAvailable":
                avail = kb / 1024 / 1024
            elif avail <= 0:
                avail = kb / 1024 / 1024
    if total > 0 and avail <= 0:
        avail = total * 0.5
    return total, avail


def _memory_macos() -> tuple[float, float]:
    """macOS：用 sysctl + vm_stat。"""
    total = 0.0
    avail = 0.0
    out = _run(["sysctl", "-n", "hw.memsize"])
    try:
        total = int(out.strip()) / _GB
    except (TypeError, ValueError):
        total = 0.0
    vm = _run(["vm_stat"])
    if vm:
        page_size = 4096
        match = re.search(r"page size of (\d+) bytes", vm)
        if match:
            page_size = int(match.group(1))
        free_pages = 0.0
        inactive_pages = 0.0
        speculative_pages = 0.0
        for line in vm.splitlines():
            if ":" not in line:
                continue
            key, _, rest = line.partition(":")
            value = rest.strip().rstrip(".")
            try:
                number = float(value)
            except ValueError:
                continue
            if key == "Pages free":
                free_pages = number
            elif key == "Pages inactive":
                inactive_pages = number
            elif key == "Pages speculative":
                speculative_pages = number
        pages = free_pages + inactive_pages + speculative_pages
        if pages > 0:
            avail = pages * page_size / _GB
    if total > 0 and avail <= 0:
        avail = total * 0.5
    return total, avail


def _memory_bsd() -> tuple[float, float]:
    """macOS 之外的 BSD 系：用 sysctl hw.physmem。"""
    total = 0.0
    out = _run(["sysctl", "-n", "hw.physmem"])
    try:
        total = int(out.strip()) / _GB
    except (TypeError, ValueError):
        total = 0.0
    return total, total * 0.5 if total > 0 else 0.0


def available_ram_gb() -> float:
    """当前可用内存（GB）。尽量用 MemAvailable / 性能计数器 / vm_stat。"""
    _, avail = _detect_memory()
    return avail


def _detect_memory() -> tuple[float, float]:
    system = platform.system().lower()
    if system == "windows":
        return _memory_windows()
    if system == "darwin":
        return _memory_macos()
    if system == "linux":
        return _memory_linux()
    if "bsd" in system or "dragonfly" in system:
        return _memory_bsd()
    # 未知平台：优先试 /proc/meminfo
    total, avail = _memory_linux()
    if total > 0:
        return total, avail
    return _memory_bsd()


# --------------------------------------------------------------------------- #
# CPU
# --------------------------------------------------------------------------- #
def _cpu_windows() -> tuple[str, int]:
    name = ""
    cores = 0
    # 优先 PowerShell CIM（Win11 新版已移除 wmic）
    ps = _run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$p=Get-CimInstance Win32_Processor | Select-Object -First 1;"
            "Write-Output $p.Name; Write-Output $p.NumberOfCores",
        ],
        timeout=12.0,
    )
    if ps:
        parts = [p.strip() for p in ps.splitlines() if p.strip()]
        if parts:
            name = parts[0]
        if len(parts) > 1:
            try:
                cores = int(parts[1])
            except ValueError:
                cores = 0

    if not name:
        out = _run(["wmic", "cpu", "get", "Name,NumberOfCores"])
        for line in (out or "").splitlines():
            line = line.strip()
            if not line or line.lower().startswith("name"):
                continue
            if "," in line:
                name_part, _, cores_part = line.partition(",")
                name = name_part.strip()
                try:
                    cores = int(cores_part.strip())
                except ValueError:
                    cores = 0
                break
            name = line
            break

    if not name:
        # 注册表兜底
        try:
            import winreg  # type: ignore  # noqa: PLC0415

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as k:
                val, _ = winreg.QueryValueEx(k, "ProcessorNameString")
                name = str(val).strip()
        except (OSError, ImportError):
            pass

    if not cores:
        cores = os.cpu_count() or 0
    return name, cores


def _cpu_linux() -> tuple[str, int]:
    name = ""
    cores = 0
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except OSError:
        content = ""
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip().lower()
        rest = rest.strip()
        if not name and key in ("model name", "hardware", "cpu model", "processor", "system cpu model"):
            name = rest
        elif not name and key == "cpu model name":
            name = rest
        if not name and key == "processor" and rest.isdigit() and platform.processor() == rest:
            name = rest
    if not name:
        name = platform.processor() or platform.machine()
    cores = os.cpu_count() or 0
    return name, cores


def _cpu_macos() -> tuple[str, int]:
    name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    cores = os.cpu_count() or 0
    return name, cores


def _detect_cpu() -> tuple[str, int]:
    system = platform.system().lower()
    if system == "windows":
        return _cpu_windows()
    if system == "darwin":
        return _cpu_macos()
    return _cpu_linux()


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
def _gpu_nvidia() -> tuple[str, float]:
    """nvidia-smi：取驱动里显存最大的那张卡。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "", 0.0
    out = _run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    best_name = ""
    best_vram = 0.0
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        name = parts[0]
        vram = 0.0
        if len(parts) > 1:
            try:
                vram = float(re.sub(r"[^\d.]", "", parts[1]) or 0)
            except ValueError:
                vram = 0.0
        if vram > best_vram:
            best_vram, best_name = vram, name
    return best_name, best_vram / 1024.0


def _gpu_rocm() -> tuple[str, float]:
    """rocm-smi：取最大显存的卡。"""
    exe = shutil.which("rocm-smi")
    if not exe:
        return "", 0.0
    out = _run([exe, "--showmeminfo", "vram", "--csv"])
    best_name = ""
    best_vram = 0.0
    for line in out.splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name = parts[0]
        try:
            vram = float(re.sub(r"[^\d.]", "", parts[-1]) or 0)
        except ValueError:
            continue
        if vram > best_vram:
            best_vram, best_name = vram, name
    return best_name, best_vram / 1024.0


def _gpu_vulkan() -> tuple[str, float]:
    """vulkaninfo：读最大显存。"""
    exe = shutil.which("vulkaninfo")
    if not exe:
        return "", 0.0
    out = _run([exe, "--summary"], timeout=12.0)
    if not out.strip():
        out = _run([exe], timeout=12.0)
    name = ""
    vram = 0.0
    for line in out.splitlines():
        low = line.lower()
        if "device name" in low:
            match = re.search(r"device name\s*[:=]\s*(.+)$", line, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
        if "heap" in low and ("memory" in low or "size" in low):
            nums = re.findall(r"(\d+(?:\.\d+)?)\s*(GiB|MiB|KiB|B)\b", line, re.IGNORECASE)
            for value, unit in nums:
                mult = {"b": 1, "kib": 1024, "mib": 1024**2, "gib": 1024**3}[unit.lower()]
                size = float(value) * mult / _GB
                if size > vram:
                    vram = size
    return name, vram


def _gpu_linux_pci() -> tuple[str, float]:
    """Linux 无专有工具时：lspci 找 VGA/3D 控制器名称（拿不到显存）。"""
    exe = shutil.which("lspci")
    if not exe:
        return "", 0.0
    out = _run([exe])
    for line in out.splitlines():
        if re.search(r"(VGA compatible controller|3D controller|Display controller)", line, re.IGNORECASE):
            return line.split(":", 2)[-1].strip(), 0.0
    return "", 0.0


def _gpu_windows_fallback() -> tuple[str, float]:
    """Windows 无 nvidia-smi 时：用 PowerShell CIM 枚举显示适配器。

    Win11 已移除 wmic，所以优先用 Get-CimInstance。要覆盖 AMD / Intel 独显与核显：
    跳过虚拟显示器（GameViewer、Parsec、Basic Display 之类）。
    """
    candidates: list[tuple[str, float]] = []

    ps = (
        "Get-CimInstance Win32_VideoController | "
        "ForEach-Object { \"$($_.Name)|$($_.AdapterRAM)|$($_.DriverVersion)\" }"
    )
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
               timeout=15.0)
    if not out:
        # 极旧系统兜底
        out2 = _run(["wmic", "path", "win32_VideoController",
                     "get", "name,AdapterRAM"], timeout=10.0)
        if out2:
            for line in out2.splitlines():
                line = line.strip()
                if not line or line.lower().startswith("name"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                name = parts[0] if parts else ""
                vram = 0.0
                if len(parts) > 1:
                    try:
                        vram = float(re.sub(r"[^\d.]", "", parts[1]) or 0) / _GB
                    except ValueError:
                        vram = 0.0
                if name:
                    candidates.append((name, vram))

    for line in (out or "").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        name = parts[0].strip()
        vram = 0.0
        if len(parts) > 1:
            try:
                vram = float(parts[1] or 0) / _GB
            except ValueError:
                vram = 0.0
        if name:
            candidates.append((name, vram))

    best_name = ""
    best_vram = 0.0
    best_score = -999.0
    for name, vram in candidates:
        low = name.lower()
        # 排除虚拟 / 基础显示适配器
        if any(k in low for k in ("virtual", "basic display", "parsec",
                                  "gameviewer", "idedisplay", "remote",
                                  "spacedesk", "oray")):
            continue
        score = vram
        if "nvidia" in low or "geforce" in low or "radeon rx" in low or "arc a" in low:
            score += 100  # 认作独显，优先
        elif "radeon" in low or "intel" in low or "uhd" in low or "iris" in low:
            score += 10
        if score > best_score:
            best_score, best_name, best_vram = score, name, vram

    return best_name, best_vram


def _vram_from_registry(name_hint: str = "") -> float:
    """AMD / Intel 的 AdapterRAM 常被截断成 4GB，用注册表 HardwareInformation.qwMemorySize 补。"""
    try:
        import winreg  # type: ignore  # noqa: PLC0415
    except ImportError:
        return 0.0

    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    best = 0.0
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
            idx = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, idx)
                except OSError:
                    break
                idx += 1
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub) as k:
                        size, _ = winreg.QueryValueEx(k, "HardwareInformation.qwMemorySize")
                        val = float(size)
                        if val > best:
                            best = val
                except OSError:
                    continue
    except OSError:
        return 0.0
    return best / _GB if best else 0.0


def _detect_gpu() -> tuple[str, float, str]:
    """返回 (名称, 显存GB, 后端)。"""
    name, vram = _gpu_nvidia()
    if name:
        return name, vram, "nvidia"

    name, vram = _gpu_rocm()
    if name:
        return name, vram, "rocm"

    if platform.system().lower() == "linux":
        name, vram = _gpu_vulkan()
        if name or vram:
            return name, vram, "vulkan"
        name, vram = _gpu_linux_pci()
        if name:
            return name, vram, "vulkan"

    if platform.system().lower() == "windows":
        name, vram = _gpu_windows_fallback()
        if name:
            return name, vram, "vulkan"

    return "", 0.0, "cpu"


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def detect() -> HardwareInfo:
    """探测本机硬件信息。任何单项失败都只留空对应字段。"""
    arch = platform.machine() or ""
    is_android = _is_android()
    system = platform.system().lower()

    try:
        cpu_model, cores = _detect_cpu()
    except Exception as exc:  # noqa: BLE001
        logger.debug("CPU 探测失败: %s", exc)
        cpu_model, cores = "", os.cpu_count() or 0

    try:
        total_ram, avail_ram = _detect_memory()
    except Exception as exc:  # noqa: BLE001
        logger.debug("内存探测失败: %s", exc)
        total_ram, avail_ram = 0.0, 0.0

    try:
        gpu_name, gpu_vram, gpu_backend = _detect_gpu()
    except Exception as exc:  # noqa: BLE001
        logger.debug("GPU 探测失败: %s", exc)
        gpu_name, gpu_vram, gpu_backend = "", 0.0, "cpu"

    if not cores:
        cores = os.cpu_count() or 0

    # AMD / Intel 的 AdapterRAM 会被截断到 4GB，用注册表补真实显存
    if gpu_name and system == "windows" and (gpu_vram <= 0 or gpu_vram >= 3.9):
        try:
            reg_vram = _vram_from_registry()
            if reg_vram > gpu_vram:
                gpu_vram = reg_vram
        except Exception as exc:  # noqa: BLE001
            logger.debug("注册表显存读取失败: %s", exc)

    return HardwareInfo(
        cpu_arch=arch,
        cpu_cores=cores,
        cpu_model_name=cpu_model,
        total_ram_gb=round(total_ram, 2),
        available_ram_gb=round(avail_ram, 2),
        gpu_name=gpu_name,
        gpu_vram_gb=round(gpu_vram, 2),
        gpu_backend=gpu_backend,
        is_arm=_is_arm(arch),
        is_android=is_android,
        extras={"system": system, "platform": platform.platform()},
    )


def summarize(hw: HardwareInfo) -> str:
    """生成一句话中文硬件摘要。"""
    parts: list[str] = []
    cpu = hw.cpu_model_name.strip() or (hw.cpu_arch or "未知 CPU")
    parts.append(f"CPU {cpu}（{hw.cpu_arch}，{hw.cpu_cores} 核）")
    if hw.total_ram_gb:
        parts.append(
            f"内存 {hw.total_ram_gb:.1f}GB（可用 {hw.available_ram_gb:.1f}GB）"
        )
    if hw.gpu_backend and hw.gpu_backend != "cpu":
        gpu = hw.gpu_name or "未知 GPU"
        if hw.gpu_vram_gb:
            parts.append(f"GPU {gpu}（{hw.gpu_backend}，{hw.gpu_vram_gb:.1f}GB 显存）")
        else:
            parts.append(f"GPU {gpu}（{hw.gpu_backend}，显存未知）")
    else:
        parts.append("无独立 GPU（纯 CPU 推理）")
    if hw.is_android:
        parts.append("Android/Termux 环境")
    elif hw.is_arm:
        parts.append("ARM 架构")
    return "；".join(parts) + "。"
