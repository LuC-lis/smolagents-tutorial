"""
公共配置：把「用哪个模型」这件事集中在一个地方。

smolagents 只有两个核心概念，先分清它们，后面就都好懂了：

    Model —— 只是「会思考的脑子」。输入一堆 messages，输出一段回复或一个工具调用。
             它不知道什么叫循环、什么叫工具，也不知道怎么执行代码。
    Agent —— 「怎么用这个脑子」。它负责：循环、决定调哪个工具、解析并真的执行
             Python 代码、把结果塞回上下文、直到给出 final_answer。

所以换模型永远只改 Model，换工作方式永远只改 Agent。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from smolagents import OpenAIModel

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")  # 读取同目录 .env 里的 SMOL_* 变量


def build_model(model_id: str | None = None, **kwargs) -> OpenAIModel:
    """返回一个 OpenAI 兼容协议的 Model。

    只要服务端实现了 OpenAI 的 /chat/completions，就能用这一套接：

        DeepSeek   SMOL_API_BASE=https://api.deepseek.com/v1
        OpenAI     SMOL_API_BASE=https://api.openai.com/v1
        OpenRouter SMOL_API_BASE=https://openrouter.ai/api/v1
        本地 Ollama SMOL_API_BASE=http://localhost:11434/v1   (key 随便填非空字符串)

    Args:
        model_id: 覆盖 .env 里的 SMOL_MODEL。
        **kwargs: 直接透传给单次补全调用，例如 temperature=0.2、max_tokens=4096。
    """
    api_key = os.getenv("SMOL_API_KEY")
    if not api_key:
        raise SystemExit(
            "缺少 SMOL_API_KEY。请复制 .env.example 为 .env 并填入你的密钥。"
        )
    return OpenAIModel(
        model_id=model_id or os.getenv("SMOL_MODEL", "deepseek-chat"),
        api_base=os.getenv("SMOL_API_BASE"),
        api_key=api_key,
        **kwargs,
    )


def rule(title: str) -> None:
    """给示例输出加个分节标题，纯装饰。"""
    print(f"\n{'=' * 8} {title} {'=' * 8}")
