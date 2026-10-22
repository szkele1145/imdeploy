"""imdeploy：本地大模型部署器。

对外提供 OpenAI 兼容 API（默认 127.0.0.1:8080），支持：

- 自动从模型站（ModelScope / HuggingFace）拉取模型目录
- 按本机 CPU / GPU / 剩余内存匹配推荐模型（0-2B 重点长上下文优化）
- 下载、校验、启动、切换模型
- OpenAI 兼容端点 + /api/{模型系列} 多路径别名
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
