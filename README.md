# ImDeploy

本地大模型部署器：按本机硬件推荐并下载 GGUF 模型，启动推理后端，对外提供 OpenAI 兼容 API。

## 功能

- **硬件自适应**：自动探测 CPU 型号 / 核心数 / 内存 / GPU 与显存，据此筛选模型
- **智能推荐**：0-2B 小模型优先优化，重点加权上下文长度（长上下文友好），并按量化格式、体积、指令微调标签综合打分
- **模型目录**：从 ModelScope 与 HuggingFace 抓取 GGUF 模型列表，带本地缓存
- **断点续传下载**：Range 续传、SHA256 校验、进度回调
- **双后端**：llama.cpp（`llama-server`）为主，兼容 Ollama
- **OpenAI 兼容 API**：`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/api/embeddings`
- **多模型别名路径**：`/api/qwen`、`/api/llama`、`/api/gemma` 等，按模型系列路由
- **中文桌面面板**：总览 / 推荐模型 / 模型库 / 在线目录 / 设置

## 安装

### Windows

下载 `imdeploy-setup-0.1.0.exe` 双击安装。安装后从开始菜单启动「ImDeploy 本地模型部署器」。

### 从源码运行

```bash
git clone <仓库地址>
cd imdeploy
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python -m imdeploy
```

## 前置：安装推理后端

模型推理需要 llama.cpp 的 `llama-server`：

1. 从 [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) 下载对应平台的构建
2. 解压到任意目录
3. 把 `llama-server`（Windows 下是 `llama-server.exe`）所在目录加入 PATH，或在面板「设置 → 推理后端 → llama-server 路径」里填写完整路径

也可以在设置里改用 Ollama：填好 `ollama_url`（默认 `http://127.0.0.1:11434`）即可。

## 使用

启动后进入桌面面板：

1. **推荐模型** → 点「按本机硬件推荐」，从列表里挑一个合适的量化版本
2. 点「下载选中模型」，等候下载完成
3. **模型库** → 选中模型 → 点「启动选中模型」
4. **总览** → 点「启动 API 服务」

## API 用法

服务默认监听 `127.0.0.1:8080`。

### OpenAI 兼容

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"你好"}]}'
```

### 按系列路由

```bash
curl http://127.0.0.1:8080/api/qwen \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
```

### 其他端点

| 端点 | 说明 |
| --- | --- |
| `GET /v1/models` | 列出已下载模型 |
| `POST /v1/completions` | 文本补全 |
| `POST /api/embeddings` | 向量化 |
| `GET /api/status` | 服务状态、已加载模型、后端情况 |
| `GET /healthz` | 健康检查 |

## 命令行

```bash
python -m imdeploy            # 打开桌面面板
python -m imdeploy --serve    # 只启动 API 服务（无界面）
python -m imdeploy --init     # 生成配置模板后退出
python -m imdeploy --cli      # 命令行帮助
```

## 目录结构

```
imdeploy/
├── src/imdeploy/
│   ├── __main__.py       入口
│   ├── config.py         配置与中文字段表
│   ├── hardware.py       CPU / 内存 / GPU 探测
│   ├── catalog.py        ModelScope / HuggingFace 目录抓取
│   ├── recommend.py      硬件匹配与打分
│   ├── downloader.py     GGUF 断点续传下载
│   ├── library.py        本地模型库
│   ├── backends.py       llama.cpp / Ollama 后端
│   ├── api_server.py     OpenAI 兼容 API
│   └── gui.py            桌面面板
├── build_tools/          打包脚本
├── tests/                测试
├── data/                 配置与模型库索引
├── models/               下载的 GGUF
└── logs/                 日志
```

## 配置文件

默认在 `data/config.yaml`，首次运行自动生成。主要节：

| 节 | 说明 |
| --- | --- |
| `app` | 监听地址、端口、访问口令、是否开浏览器 |
| `backend` | 后端偏好、llama-server 路径、Ollama 地址、GPU 层数、线程数、上下文长度 |
| `catalog` | 目录抓取关键词与缓存 |
| `recommend` | 推荐策略（小模型偏好、长上下文加分、内存余量） |
| `download` | 超时、校验、续传 |
| `api` | 模型别名映射 |

## 说明

- 所有数据留在本机，模型下载走你配置的模型站
- 端口默认只监听 `127.0.0.1`；如需局域网访问请改 `app.host` 并设置 `app.token`
