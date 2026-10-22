"""桌面 GUI：本地模型部署器的图形控制台。

页签：

- **总览**：服务状态、当前加载模型、硬件信息、API 端点
- **推荐**：按本机硬件匹配的推荐模型（可一键下载）
- **模型库**：已下载模型、启动/停止/删除
- **目录**：在线抓取模型目录、搜索
- **设置**：服务端口、后端、下载参数
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import ttk
from typing import Any

from .config import Config, field_label, section_title

# ---------------------------------------------------------------- 配色

BG = "#0d1117"
PANEL = "#161b22"
PANEL2 = "#1c2129"
BORDER = "#30363d"
TEXT = "#e6edf3"
DIM = "#8b949e"
ACCENT = "#58a6ff"
OK = "#3fb950"
WARN = "#d29922"
ERR = "#f85149"


class _Task:
    """后台任务结果回传。"""

    def __init__(self) -> None:
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def put(self, kind: str, payload: Any) -> None:
        self.queue.put((kind, payload))

    def drain(self) -> list[tuple[str, Any]]:
        out = []
        while True:
            try:
                out.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return out


class ControlPanelGUI:
    """本地模型部署器桌面面板。"""

    def __init__(self, cfg: Config | None = None, config_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path else Path("data/config.yaml")
        self.cfg = cfg or Config.load(self.config_path)
        self.tasks = _Task()
        self.library = None
        self.hardware = None
        self.catalog: list = []
        self.recommendations: list = []
        self.active_model = None
        self.server = None
        self._vars: dict[str, Any] = {}
        self._masked: dict[str, str] = {}

        self.root = tk.Tk()
        self.root.title("本地模型部署器")
        self.root.geometry("900x680")
        self.root.minsize(820, 600)
        self.root.configure(bg=BG)
        self._style()
        self._build()
        self._poll()

    # ------------------------------------------------------------ 样式

    def _style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:  # noqa: BLE001
            pass
        style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL2,
                        bordercolor=BORDER, font=("Microsoft YaHei UI", 10))
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=DIM,
                        padding=(18, 9), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", BG), ("active", PANEL2)],
                  foreground=[("selected", ACCENT), ("active", TEXT)])
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Hint.TLabel", background=BG, foreground=DIM,
                        font=("Microsoft YaHei UI", 9))
        style.configure("Title.TLabel", background=BG, foreground=TEXT,
                        font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("TButton", background=PANEL2, foreground=TEXT,
                        padding=(13, 7), relief="flat", borderwidth=1)
        style.map("TButton", background=[("active", BORDER)], foreground=[("active", ACCENT)])
        style.configure("Accent.TButton", background=ACCENT, foreground="#06101f",
                        padding=(15, 9), font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#79b8ff")])
        style.configure("Danger.TButton", background=ERR, foreground="#fff",
                        padding=(13, 7), font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("TCheckbutton", background=BG, foreground=TEXT)
        style.configure("TEntry", fieldbackground=PANEL2, foreground=TEXT,
                        insertcolor=TEXT, bordercolor=BORDER)
        style.configure("TCombobox", fieldbackground=PANEL2, foreground=TEXT,
                        bordercolor=BORDER)
        style.configure("Treeview", background=PANEL2, fieldbackground=PANEL2,
                        foreground=TEXT, borderwidth=0, rowheight=26)
        style.configure("Treeview.Heading", background=PANEL, foreground=DIM,
                        font=("Microsoft YaHei UI", 9, "bold"))
        style.map("Treeview", background=[("selected", ACCENT)],
                  foreground=[("selected", "#06101f")])

    # ------------------------------------------------------------ 构建

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=PANEL, highlightthickness=1,
                          highlightbackground=BORDER)
        header.pack(fill="x", side="top")
        tk.Frame(header, bg=BORDER, height=1).pack(fill="x", side="bottom")

        tk.Label(header, text="🧠 本地模型部署器", bg=PANEL, fg=TEXT,
                 font=("Microsoft YaHei UI", 13, "bold")).pack(side="left", padx=18, pady=12)

        self.lbl_service = tk.Label(header, text="服务: 未启动", bg=PANEL, fg=DIM,
                                    font=("Microsoft YaHei UI", 10))
        self.lbl_service.pack(side="left", padx=8)

        self.lbl_model = tk.Label(header, text="模型: -", bg=PANEL, fg=DIM,
                                  font=("Microsoft YaHei UI", 10))
        self.lbl_model.pack(side="left", padx=8)

        btns = tk.Frame(header, bg=PANEL)
        btns.pack(side="right", padx=16)
        self.btn_service = ttk.Button(btns, text="启动 API 服务", style="Accent.TButton",
                                      command=self._toggle_service)
        self.btn_service.pack(side="left")
        ttk.Button(btns, text="打开文档", command=self._open_docs).pack(side="left", padx=(8, 0))

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_overview = ttk.Frame(self.nb)
        self.tab_recommend = ttk.Frame(self.nb)
        self.tab_library = ttk.Frame(self.nb)
        self.tab_catalog = ttk.Frame(self.nb)
        self.tab_settings = ttk.Frame(self.nb)

        self.nb.add(self.tab_overview, text="  总览  ")
        self.nb.add(self.tab_recommend, text="  推荐模型  ")
        self.nb.add(self.tab_library, text="  模型库  ")
        self.nb.add(self.tab_catalog, text="  在线目录  ")
        self.nb.add(self.tab_settings, text="  设置  ")

        self._build_overview()
        self._build_recommend()
        self._build_library()
        self._build_catalog()
        self._build_settings()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------ 总览

    def _build_overview(self) -> None:
        wrap = tk.Frame(self.tab_overview, bg=BG)
        wrap.pack(fill="both", expand=True, padx=16, pady=14)

        tk.Label(wrap, text="服务状态", bg=BG, fg=ACCENT,
                 font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        self.ov_service = tk.Label(wrap, text="加载中…", bg=BG, fg=DIM,
                                   justify="left", anchor="w", font=("Microsoft YaHei UI", 10))
        self.ov_service.pack(anchor="w", pady=(4, 12))

        tk.Label(wrap, text="本机硬件", bg=BG, fg=ACCENT,
                 font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        self.ov_hw = tk.Label(wrap, text="检测中…", bg=BG, fg=DIM,
                              justify="left", anchor="w", font=("Microsoft YaHei UI", 10))
        self.ov_hw.pack(anchor="w", pady=(4, 12))

        tk.Label(wrap, text="接口端点", bg=BG, fg=ACCENT,
                 font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        self.ov_api = tk.Label(wrap, text="", bg=BG, fg=DIM, justify="left",
                               anchor="w", font=("Microsoft YaHei UI", 10))
        self.ov_api.pack(anchor="w", pady=(4, 12))

        row = tk.Frame(wrap, bg=BG)
        row.pack(fill="x", pady=6)
        ttk.Button(row, text="重新检测硬件", command=self._detect_hw).pack(side="left")
        ttk.Button(row, text="刷新状态", command=self._refresh_overview).pack(side="left", padx=8)
        ttk.Button(row, text="复制调用示例", command=self._copy_example).pack(side="left")

        # 推理运行库（llama.cpp）—— 懒人版一键安装
        tk.Label(wrap, text="推理运行库", bg=BG, fg=ACCENT,
                 font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(14, 0))
        self.ov_runtime = tk.Label(wrap, text="检测中…", bg=BG, fg=DIM, justify="left",
                                   anchor="w", font=("Microsoft YaHei UI", 10))
        self.ov_runtime.pack(anchor="w", pady=(4, 6))

        rrow = tk.Frame(wrap, bg=BG)
        rrow.pack(fill="x")
        self.btn_install_runtime = ttk.Button(
            rrow, text="一键安装推理运行库", style="Accent.TButton",
            command=self._install_runtime)
        self.btn_install_runtime.pack(side="left")
        ttk.Button(rrow, text="重新检测", command=self._refresh_runtime).pack(side="left", padx=8)

    def _refresh_runtime(self) -> None:
        """刷新 llama-server 检测状态。"""
        try:
            from .backends import detect_backends  # noqa: PLC0415

            backends = detect_backends(self.cfg)
            llama = next((b for b in backends if getattr(b, "name", "") == "llama.cpp"), None)
            if llama is not None and llama.is_available():
                path = getattr(llama, "server_path", "") or ""
                self.ov_runtime.config(text=f"已就绪：{path}", fg=OK)
                self.btn_install_runtime.config(text="重新安装运行库")
            else:
                self.ov_runtime.config(
                    text="未安装 —— 点右边按钮自动下载安装，无需手动配置 PATH", fg=WARN)
                self.btn_install_runtime.config(text="一键安装推理运行库")
        except Exception as exc:  # noqa: BLE001
            self.ov_runtime.config(text=f"检测失败：{exc}", fg=ERR)

    def _install_runtime(self) -> None:
        if getattr(self, "_installing", False):
            self._toast("正在安装中，请稍候…")
            return
        self._installing = True
        self.btn_install_runtime.config(state="disabled", text="正在安装…")
        self.ov_runtime.config(text="正在查询最新版本…", fg=DIM)
        threading.Thread(target=self._install_runtime_worker, daemon=True).start()

    def _install_runtime_worker(self) -> None:
        try:
            from .runtime_install import install_llama_runtime  # noqa: PLC0415

            proxy = str(self.cfg.get("app.proxy", "") or "") or None

            def on_progress(done: int, total: int) -> None:
                if total > 0:
                    pct = done * 100.0 / total
                    self.tasks.put("runtime_progress",
                                   f"下载中 {pct:.0f}%（{done / 1048576:.1f}/"
                                   f"{total / 1048576:.1f} MB）")
                else:
                    self.tasks.put("runtime_progress", f"下载中 {done / 1048576:.1f} MB")

            def on_log(msg: str) -> None:
                self.tasks.put("runtime_progress", msg)

            info = install_llama_runtime(
                self.cfg.base_dir / "bin",
                proxy=proxy,
                progress=on_progress,
                log=on_log,
            )
            self.tasks.put("runtime_done", info)
        except Exception as exc:  # noqa: BLE001
            self.tasks.put("runtime_error", f"安装出错：{exc}")

    def _refresh_overview(self) -> None:
        host = str(self.cfg.get("app.host", "127.0.0.1"))
        port = int(self.cfg.get("app.port", 8080))
        base = f"http://{host}:{port}"
        running = self.server is not None
        self.lbl_service.config(text="服务: 运行中" if running else "服务: 未启动",
                                fg=OK if running else DIM)
        self.ov_service.config(
            text=f"状态：{'运行中' if running else '未启动'}\n"
                 f"监听：{base}\n"
                 f"请求口令：{'已设置' if self.cfg.get('app.token') else '未设置（不校验）'}"
        )
        if self.hardware:
            from .hardware import summarize_hw  # type: ignore  # noqa: PLC0415
            self.ov_hw.config(text=summarize_hw(self.hardware))
        self.ov_api.config(
            text=f"OpenAI 兼容：{base}/v1/chat/completions\n"
                 f"模型列表：{base}/v1/models\n"
                 f"多模型别名：{base}/api/qwen、/api/llama、/api/gemma …\n"
                 f"状态查询：{base}/api/status"
        )

    def _copy_example(self) -> None:
        host = str(self.cfg.get("app.host", "127.0.0.1"))
        port = int(self.cfg.get("app.port", 8080))
        snippet = (
            f"curl {host}:{port}/v1/chat/completions \\\n"
            f'  -H "Content-Type: application/json" \\\n'
            f'  -d \'{{"model":"qwen","messages":[{{"role":"user","content":"你好"}}]}}\''
        )
        self.root.clipboard_clear()
        self.root.clipboard_append(snippet)
        self._toast("调用示例已复制到剪贴板")

    # ------------------------------------------------------------ 推荐

    def _build_recommend(self) -> None:
        wrap = tk.Frame(self.tab_recommend, bg=BG)
        wrap.pack(fill="both", expand=True, padx=16, pady=14)

        bar = tk.Frame(wrap, bg=BG)
        bar.pack(fill="x")
        self.var_ctx = tk.StringVar(value="32768")
        ttk.Label(bar, text="期望上下文长度：").pack(side="left")
        cb = ttk.Combobox(bar, textvariable=self.var_ctx, width=10, state="readonly",
                          values=["4096", "8192", "16384", "32768", "65536", "131072"])
        cb.pack(side="left")
        ttk.Button(bar, text="按本机硬件推荐", style="Accent.TButton",
                   command=self._recommend).pack(side="left", padx=10)
        ttk.Button(bar, text="抓取在线目录", command=self._fetch_catalog).pack(side="left")

        self.rec_tree = ttk.Treeview(
            wrap, columns=("tier", "name", "params", "ctx", "quant", "size", "score"),
            show="headings", height=16)
        heads = {
            "tier": ("推荐档位", 90), "name": ("模型", 300), "params": ("参数量", 80),
            "ctx": ("上下文", 90), "quant": ("量化", 100), "size": ("体积", 80),
            "score": ("评分", 60),
        }
        for key, (text, width) in heads.items():
            self.rec_tree.heading(key, text=text)
            self.rec_tree.column(key, width=width, anchor="w")
        self.rec_tree.pack(fill="both", expand=True, pady=(10, 6))

        row = tk.Frame(wrap, bg=BG)
        row.pack(fill="x")
        ttk.Button(row, text="下载选中模型", style="Accent.TButton",
                   command=self._download_selected).pack(side="left")
        ttk.Button(row, text="下载并启动", command=self._download_and_start).pack(side="left", padx=8)
        self.lbl_rec_status = ttk.Label(row, text="", style="Hint.TLabel")
        self.lbl_rec_status.pack(side="left", padx=8)

    def _recommend(self) -> None:
        self.lbl_rec_status.config(text="正在按硬件匹配…")
        threading.Thread(target=self._recommend_worker, daemon=True).start()

    def _recommend_worker(self) -> None:
        try:
            from .catalog import build_catalog  # type: ignore  # noqa: PLC0415
            from .hardware import detect  # type: ignore  # noqa: PLC0415
            from .recommend import recommend  # type: ignore  # noqa: PLC0415

            hw = self.hardware or detect()
            self.hardware = hw
            if not self.catalog:
                try:
                    self.catalog = build_catalog(self.cfg.get("catalog.queries", []),
                                                 self.cfg.get("catalog.limit_per_query", 30))
                except Exception:  # noqa: BLE001
                    self.catalog = []
            ctx = int(self.var_ctx.get() or 8192)
            self.recommendations = recommend(self.catalog, hw, limit=60,
                                             min_ctx=int(self.cfg.get("catalog.min_context", 8192)))
            self.tasks.put("recommend", self.recommendations)
        except Exception as exc:  # noqa: BLE001
            self.tasks.put("error", f"推荐失败: {exc}")

    def _download_selected(self) -> None:
        sel = self.rec_tree.selection()
        if not sel:
            self._toast("请先选中一个模型")
            return
        self._download_by_id(sel[0], start_after=False)

    def _download_and_start(self) -> None:
        sel = self.rec_tree.selection()
        if not sel:
            self._toast("请先选中一个模型")
            return
        self._download_by_id(sel[0], start_after=True)

    def _download_by_id(self, mid: str, start_after: bool) -> None:
        rec = next((r for r in self.recommendations if r.entry.id == mid), None)
        if rec is None:
            self._toast("找不到该模型")
            return
        threading.Thread(target=self._download_worker, args=(rec, start_after),
                         daemon=True).start()

    def _download_worker(self, rec, start_after: bool) -> None:
        try:
            from .downloader import Downloader, resolve_url  # noqa: PLC0415

            entry = rec.entry
            dest = self.cfg.models_dir / f"{entry.id}.gguf"
            url = entry.download_url or resolve_url(entry.repo, entry.filename,
                                                    entry.source)
            dl = Downloader(timeout=int(self.cfg.get("download.timeout", 600)),
                            resume=bool(self.cfg.get("download.resume", True)))

            def progress(task):
                self.tasks.put("progress", task)

            task = dl.download(url, dest, progress, entry.sha256)
            if task.status == "done":
                self.library.register(dest, {
                    "family": entry.family, "params_b": entry.params_b,
                    "quant": entry.quant, "ctx_len": entry.ctx_len,
                    "description": entry.description, "download_url": url,
                    "source": entry.source,
                })
                self.tasks.put("done", (entry.name, start_after))
            else:
                self.tasks.put("error", f"下载失败: {task.error}")
        except Exception as exc:  # noqa: BLE001
            self.tasks.put("error", f"下载异常: {exc}")

    # ------------------------------------------------------------ 模型库

    def _build_library(self) -> None:
        wrap = tk.Frame(self.tab_library, bg=BG)
        wrap.pack(fill="both", expand=True, padx=16, pady=14)

        bar = tk.Frame(wrap, bg=BG)
        bar.pack(fill="x")
        ttk.Button(bar, text="扫描目录", command=self._scan_library).pack(side="left")
        ttk.Button(bar, text="启动选中模型", style="Accent.TButton",
                   command=self._start_selected_model).pack(side="left", padx=8)
        ttk.Button(bar, text="停止模型", command=self._stop_model).pack(side="left")
        ttk.Button(bar, text="删除选中", style="Danger.TButton",
                   command=self._delete_selected).pack(side="left", padx=8)

        cols = ("name", "family", "params", "ctx", "quant", "size")
        self.lib_tree = ttk.Treeview(wrap, columns=cols, show="headings", height=18)
        heads = {
            "name": ("模型名称", 330), "family": ("系列", 90), "params": ("参数量", 90),
            "ctx": ("上下文", 100), "quant": ("量化", 110), "size": ("体积", 80),
        }
        for key, (text, width) in heads.items():
            self.lib_tree.heading(key, text=text)
            self.lib_tree.column(key, width=width, anchor="w")
        self.lib_tree.pack(fill="both", expand=True, pady=(10, 0))

    def _scan_library(self) -> None:
        if self.library is None:
            from .library import ModelLibrary  # noqa: PLC0415
            self.library = ModelLibrary(self.cfg)
        models = self.library.scan()
        self._fill_lib_tree(models)
        self._toast(f"扫描完成，共 {len(models)} 个模型")

    def _fill_lib_tree(self, models) -> None:
        for item in self.lib_tree.get_children():
            self.lib_tree.delete(item)
        for m in models:
            self.lib_tree.insert("", "end", iid=m.id, values=(
                m.name, m.family, f"{m.params_b}B" if m.params_b else "-",
                f"{m.ctx_len // 1024}K" if m.ctx_len else "-",
                m.quant or "-", f"{m.size_gb}GB",
            ))

    def _start_selected_model(self) -> None:
        sel = self.lib_tree.selection()
        if not sel:
            self._toast("请先选中一个模型")
            return
        threading.Thread(target=self._start_model_worker, args=(sel[0],), daemon=True).start()

    def _start_model_worker(self, mid: str) -> None:
        try:
            from .backends import detect_backends  # noqa: PLC0415

            backends = detect_backends(self.cfg)
            if not backends:
                self.tasks.put("error", "没有可用的推理后端（llama-server 未找到？）")
                return
            if self.library is None:
                from .library import ModelLibrary  # noqa: PLC0415

                self.library = ModelLibrary(self.cfg)
            model = self.library.get(mid)
            if model is None:
                self.tasks.put("error", "模型不存在")
                return
            backend = next((b for b in backends if b.is_available()), backends[0])
            info = backend.start_model(
                mid,
                gguf=Path(model.path),
                ctx_size=int(self.cfg.get("backend.context_size", 8192)),
            )
            self.tasks.put("started", (mid, info))
        except Exception as exc:  # noqa: BLE001
            self.tasks.put("error", f"启动失败: {exc}")

    def _stop_model(self) -> None:
        threading.Thread(target=self._stop_model_worker, daemon=True).start()

    def _stop_model_worker(self) -> None:
        try:
            from .backends import detect_backends  # type: ignore  # noqa: PLC0415

            for b in detect_backends(self.cfg):
                b.stop_model()
            self.tasks.put("stopped", None)
        except Exception as exc:  # noqa: BLE001
            self.tasks.put("error", f"停止失败: {exc}")

    def _delete_selected(self) -> None:
        sel = self.lib_tree.selection()
        if not sel:
            self._toast("请先选中一个模型")
            return
        if self.library is None:
            from .library import ModelLibrary  # noqa: PLC0415
            self.library = ModelLibrary(self.cfg)
        if self.library.remove(sel[0]):
            self._scan_library()
            self._toast("已删除")
        else:
            self._toast("删除失败")

    # ------------------------------------------------------------ 在线目录

    def _build_catalog(self) -> None:
        wrap = tk.Frame(self.tab_catalog, bg=BG)
        wrap.pack(fill="both", expand=True, padx=16, pady=14)

        bar = tk.Frame(wrap, bg=BG)
        bar.pack(fill="x")
        self.var_search = tk.StringVar()
        entry = ttk.Entry(bar, textvariable=self.var_search)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _e: self._search_catalog())
        ttk.Button(bar, text="搜索", style="Accent.TButton",
                   command=self._search_catalog).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="抓取默认目录", command=self._fetch_catalog).pack(side="left", padx=8)

        self.cat_tree = ttk.Treeview(
            wrap, columns=("name", "family", "params", "ctx", "quant", "size", "src"),
            show="headings", height=18)
        heads = {
            "name": ("模型", 320), "family": ("系列", 90), "params": ("参数量", 90),
            "ctx": ("上下文", 100), "quant": ("量化", 110), "size": ("体积", 80),
            "src": ("来源", 100),
        }
        for key, (text, width) in heads.items():
            self.cat_tree.heading(key, text=text)
            self.cat_tree.column(key, width=width, anchor="w")
        self.cat_tree.pack(fill="both", expand=True, pady=(10, 0))

        row = tk.Frame(wrap, bg=BG)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="下载选中", style="Accent.TButton",
                   command=self._download_from_catalog).pack(side="left")

    def _fetch_catalog(self) -> None:
        self._toast("正在从模型站抓取目录…")
        threading.Thread(target=self._catalog_worker, daemon=True).start()

    def _catalog_worker(self) -> None:
        try:
            from .catalog import build_catalog  # type: ignore  # noqa: PLC0415

            self.catalog = build_catalog(self.cfg.get("catalog.queries", []),
                                         int(self.cfg.get("catalog.limit_per_query", 30)))
            self.tasks.put("catalog", self.catalog)
        except Exception as exc:  # noqa: BLE001
            self.tasks.put("error", f"抓取失败: {exc}")

    def _search_catalog(self) -> None:
        kw = self.var_search.get().strip()
        if not kw:
            self._fetch_catalog()
            return
        threading.Thread(target=self._search_worker, args=(kw,), daemon=True).start()

    def _search_worker(self, kw: str) -> None:
        try:
            from .catalog import fetch_hf_models, fetch_modelscope_models  # type: ignore  # noqa: PLC0415

            results = []
            results.extend(fetch_modelscope_models(kw, 30))
            results.extend(fetch_hf_models(kw, 30))
            self.tasks.put("catalog", results)
        except Exception as exc:  # noqa: BLE001
            self.tasks.put("error", f"搜索失败: {exc}")

    def _download_from_catalog(self) -> None:
        sel = self.cat_tree.selection()
        if not sel:
            self._toast("请先选中一个模型")
            return
        entry = next((e for e in self.catalog if e.id == sel[0]), None)
        if entry is None:
            self._toast("找不到该模型")
            return
        threading.Thread(target=self._download_worker, args=(entry, False), daemon=True).start()

    # ------------------------------------------------------------ 设置

    def _build_settings(self) -> None:
        import tkinter as tk  # noqa: PLC0415

        wrap = tk.Frame(self.tab_settings, bg=BG)
        wrap.pack(fill="both", expand=True, padx=16, pady=14)

        self.form = tk.Frame(wrap, bg=BG)
        self.form.pack(fill="both", expand=True)

        bar = tk.Frame(wrap, bg=BG)
        bar.pack(fill="x", pady=(10, 0))
        ttk.Button(bar, text="保存设置", style="Accent.TButton",
                   command=self._save_settings).pack(side="left", fill="x", expand=True)
        ttk.Button(bar, text="刷新", command=self._load_settings).pack(side="left", fill="x",
                                                                     expand=True, padx=(8, 0))

        self._load_settings()

    def _load_settings(self) -> None:
        import tkinter as tk  # noqa: PLC0415

        for child in self.form.winfo_children():
            child.destroy()
        self._vars.clear()

        groups = [
            ("服务设置", ["app.host", "app.port", "app.token", "app.open_browser"]),
            ("推理后端", ["backend.prefer", "backend.llama_server", "backend.ollama_url",
                          "backend.base_port", "backend.gpu_layers", "backend.threads",
                          "backend.context_size", "backend.max_context_size"]),
            ("推荐策略", ["recommend.prefer_small", "recommend.small_params_max",
                          "recommend.long_context_bonus", "recommend.memory_headroom"]),
            ("下载设置", ["download.timeout", "download.verify_sha256", "download.resume"]),
        ]
        for title, keys in groups:
            tk.Label(self.form, text=title, bg=BG, fg=ACCENT,
                     font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w", pady=(10, 4))
            for key in keys:
                self._add_field(key, self.cfg.get(key))

    def _add_field(self, dotted: str, value: Any) -> None:
        import tkinter as tk  # noqa: PLC0415

        row = tk.Frame(self.form, bg=BG)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=field_label(dotted), bg=BG, fg=TEXT, width=24,
                 anchor="w", font=("Microsoft YaHei UI", 10)).pack(side="left")

        if isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            ttk.Checkbutton(row, variable=var).pack(side="left")
            self._vars[dotted] = var
            return

        var = tk.StringVar(value="" if value is None else str(value))
        secret = "token" in dotted or "password" in dotted
        ttk.Entry(row, textvariable=var, width=40,
                  show="•" if secret else "").pack(side="left", fill="x", expand=True)
        self._vars[dotted] = (var, "number" if isinstance(value, int) and
                              not isinstance(value, bool) else "text")

    def _save_settings(self) -> None:
        changed = []
        for dotted, holder in self._vars.items():
            if isinstance(holder, tk.BooleanVar):
                new = holder.get()
            else:
                var, kind = holder
                raw = var.get()
                if kind == "number":
                    try:
                        new = int(raw)
                    except ValueError:
                        new = raw
                else:
                    new = raw
            if self.cfg.get(dotted) != new:
                self.cfg.set(dotted, new)
                changed.append(dotted)
        if changed:
            self.cfg.save(self.config_path)
            self._toast(f"已保存 {len(changed)} 项")
        else:
            self._toast("没有改动")

    # ------------------------------------------------------------ 服务

    def _toggle_service(self) -> None:
        if self.server is not None:
            self._stop_service()
        else:
            self._start_service()

    def _start_service(self) -> None:
        try:
            from .api_server import build_server  # noqa: PLC0415

            httpd, api = build_server(self.cfg, backends=None)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            self.server = httpd
            self.api = api
            self.btn_service.config(text="停止 API 服务")
            self._toast(f"API 服务已启动：http://{api.host}:{api.port}")
        except Exception as exc:  # noqa: BLE001
            self.server = None
            self._toast(f"启动失败: {exc}")
        self._refresh_overview()

    def _stop_service(self) -> None:
        try:
            if self.server is not None:
                self.server.shutdown()
                self.server.server_close()
        except Exception:  # noqa: BLE001
            pass
        self.server = None
        self.btn_service.config(text="启动 API 服务")
        self._toast("API 服务已停止")
        self._refresh_overview()

    def _open_docs(self) -> None:
        host = str(self.cfg.get("app.host", "127.0.0.1"))
        port = int(self.cfg.get("app.port", 8080))
        webbrowser.open(f"http://{host}:{port}/docs")

    # ------------------------------------------------------------ 轮询

    def _poll(self) -> None:
        try:
            self._handle_tasks()
            self._refresh_overview()
            if not getattr(self, "_runtime_checked", False):
                self._runtime_checked = True
                self._refresh_runtime()
        except Exception:  # noqa: BLE001
            pass
        self.root.after(1500, self._poll)

    def _handle_tasks(self) -> None:
        for kind, payload in self.tasks.drain():
            if kind == "recommend":
                self._fill_rec_tree(payload)
                self.lbl_rec_status.config(text=f"共 {len(payload)} 个候选")
            elif kind == "catalog":
                self._fill_cat_tree(payload)
                self.lbl_rec_status.config(text=f"目录 {len(payload)} 条")
            elif kind == "progress":
                self.lbl_rec_status.config(
                    text=f"下载中 {payload.percent:.1f}% ({payload.speed_mb:.1f} MB)")
            elif kind == "done":
                name, start_after = payload
                self.lbl_rec_status.config(text=f"{name} 下载完成")
                self._toast(f"{name} 已下载")
                if start_after:
                    self._scan_library()
            elif kind == "started":
                mid, info = payload
                self.active_model = mid
                self.lbl_model.config(text=f"模型: {mid}")
                self._toast(f"模型 {mid} 已启动")
            elif kind == "stopped":
                self.active_model = None
                self.lbl_model.config(text="模型: -")
                self._toast("模型已停止")
            elif kind == "error":
                self.lbl_rec_status.config(text=str(payload))
                self._toast(str(payload))
            elif kind == "runtime_progress":
                self.ov_runtime.config(text=str(payload), fg=DIM)
            elif kind == "runtime_done":
                info = payload
                self._installing = False
                self.btn_install_runtime.config(state="normal")
                if info.ok:
                    self.cfg.set("backend.llama_server", info.exe_path)
                    self.cfg.save(self.config_path)
                    self.ov_runtime.config(
                        text=f"已就绪：{info.exe_path}", fg=OK)
                    self.btn_install_runtime.config(text="重新安装运行库")
                    self._toast(info.message)
                else:
                    self.ov_runtime.config(text=info.message, fg=ERR)
                    self._toast(info.message)
            elif kind == "runtime_error":
                self._installing = False
                self.btn_install_runtime.config(state="normal")
                self.ov_runtime.config(text=str(payload), fg=ERR)
                self._toast(str(payload))

    def _fill_rec_tree(self, recs) -> None:
        for item in self.rec_tree.get_children():
            self.rec_tree.delete(item)
        for r in recs:
            e = r.entry
            self.rec_tree.insert("", "end", iid=e.id, values=(
                r.tier, e.name, f"{e.params_b}B" if e.params_b else "-",
                f"{e.ctx_len // 1024}K" if e.ctx_len else "-",
                e.quant or "-", f"{e.size_gb}GB" if e.size_gb else "-",
                round(r.score),
            ))

    def _fill_cat_tree(self, entries) -> None:
        for item in self.cat_tree.get_children():
            self.cat_tree.delete(item)
        for e in entries:
            self.cat_tree.insert("", "end", iid=e.id, values=(
                e.name, e.family, f"{e.params_b}B" if e.params_b else "-",
                f"{e.ctx_len // 1024}K" if e.ctx_len else "-",
                e.quant or "-", f"{e.size_gb}GB" if e.size_gb else "-",
                e.source,
            ))

    # ------------------------------------------------------------ 杂项

    def _detect_hw(self) -> None:
        def worker():
            try:
                from .hardware import detect  # type: ignore  # noqa: PLC0415
                self.hardware = detect()
                self.tasks.put("hw", self.hardware)
            except Exception as exc:  # noqa: BLE001
                self.tasks.put("error", f"硬件检测失败: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _toast(self, message: str) -> None:
        toast = tk.Label(self.root, text=message, bg=PANEL2, fg=TEXT,
                         font=("Microsoft YaHei UI", 10), padx=16, pady=10,
                         highlightthickness=1, highlightbackground=BORDER)
        toast.place(relx=0.5, rely=1.0, anchor="s", y=-20)
        self.root.after(2800, toast.destroy)

    def _on_close(self) -> None:
        try:
            if self.server is not None:
                self.server.shutdown()
                self.server.server_close()
        except Exception:  # noqa: BLE001
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_gui(config_path: str | Path | None = None) -> int:
    """桌面 GUI 入口。"""
    gui = ControlPanelGUI(config_path=config_path)
    gui.run()
    return 0
