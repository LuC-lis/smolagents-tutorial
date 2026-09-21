"""
smol —— 你自己的 smolagents 命令行入口。

比官方 `smolagent` 命令多做了三件事：
    * 自动从当前目录的 .env 读 SMOL_API_KEY / SMOL_API_BASE / SMOL_MODEL，
      不用每次敲 --api-base / --api-key
    * 默认挂上可用的内置工具（Google 搜索 / 维基 / 抓网页）
    * 不带参数时进入多轮对话模式，而不是走官方那套交互式问卷

用法：
    uv run smol "2 的 100 次方有多少位数字？"
    uv run smol                                        # 进入多轮对话
    uv run smol --plan --steps 12 "调研一下 xxx 并给我来源"
    uv run smol -t web_search wikipedia_search "..."   # 指定工具
    uv run smol --model deepseek-v4-pro "..."          # 换模型
    uv run smol -v 2 "..."                             # 打印每步的代码
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    OpenAIModel,
    PythonInterpreterTool,
    VisitWebpageTool,
    WikipediaSearchTool,
)

__all__ = ["build_model", "build_agent", "main", "TOOL_FACTORIES"]


# ---------------------------------------------------------------- 模型

def build_model(model_id: str | None = None, **kwargs) -> OpenAIModel:
    """按 .env 里的配置造一个 OpenAI 兼容的 Model。

    .env 需要三个变量（见 .env.example）：
        SMOL_API_KEY    你的密钥
        SMOL_API_BASE   兼容 OpenAI 协议的 base URL
        SMOL_MODEL      模型 id
    """
    # 先找当前工作目录的 .env（用户在哪跑就在哪读），再退回环境变量。
    load_dotenv(Path.cwd() / ".env")

    api_key = os.getenv("SMOL_API_KEY")
    if not api_key:
        sys.exit(
            "缺少 SMOL_API_KEY。\n"
            "请复制 .env.example 为 .env，并填入你的密钥与 base URL。"
        )

    return OpenAIModel(
        model_id=model_id or os.getenv("SMOL_MODEL", "deepseek-chat"),
        api_base=os.getenv("SMOL_API_BASE"),
        api_key=api_key,
        **kwargs,
    )


# ---------------------------------------------------------------- 工具

# 名字 -> 构造函数。想加自己的工具，往这里加一行就行。
TOOL_FACTORIES = {
    "web_search": GoogleSearchTool,          # 需要 SERPAPI_API_KEY
    "wikipedia_search": WikipediaSearchTool,
    "visit_webpage": VisitWebpageTool,
    "python_interpreter": PythonInterpreterTool,
}

DEFAULT_TOOLS = ["web_search", "wikipedia_search", "visit_webpage"]


def resolve_tools(names: list[str]) -> list:
    tools = []
    for name in names:
        factory = TOOL_FACTORIES.get(name)
        if factory is None:
            sys.exit(f"未知工具 '{name}'。可用：{', '.join(TOOL_FACTORIES)}")
        try:
            tools.append(factory())
        except Exception as exc:  # 缺 key、访问不通等
            print(f"[warn] 跳过工具 {name}：{exc}", file=sys.stderr)
    return tools


# ---------------------------------------------------------------- Agent

def build_agent(
    model_id: str | None = None,
    tools: list[str] | None = None,
    verbosity: int = 1,
    max_steps: int = 10,
    plan: int | None = None,
    imports: list[str] | None = None,
    stream: bool = False,
) -> CodeAgent:
    return CodeAgent(
        tools=resolve_tools(tools if tools is not None else DEFAULT_TOOLS),
        model=build_model(model_id),
        max_steps=max_steps,
        planning_interval=plan,
        additional_authorized_imports=imports or [],
        stream_outputs=stream,
        verbosity_level=verbosity,
    )


# ---------------------------------------------------------------- CLI

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="smol",
        description="用 DeepSeek / 任意 OpenAI 兼容模型驱动的 CodeAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("task", nargs="*", help="要交给 Agent 的任务；留空则进入多轮对话模式")
    p.add_argument("-m", "--model", help="覆盖 .env 里的 SMOL_MODEL")
    p.add_argument(
        "-t", "--tools", nargs="*", default=None,
        help=f"启用的工具，空格分隔。可用：{', '.join(TOOL_FACTORIES)}（默认 {len(DEFAULT_TOOLS)} 个）",
    )
    p.add_argument("--no-tools", action="store_true", help="不带任何工具（纯推理 + 计算）")
    p.add_argument("-v", "--verbosity", type=int, default=1, choices=[0, 1, 2], help="日志级别，默认 1")
    p.add_argument("-s", "--steps", type=int, default=10, help="最大步数，默认 10")
    p.add_argument("-p", "--plan", type=int, nargs="?", const=3, help="每 N 步重新规划一次（默认 3）")
    p.add_argument("-i", "--imports", nargs="*", default=None, help="额外放行的 import，如 csv json")
    p.add_argument("--stream", action="store_true", help="流式打印模型的输出增量")
    return p.parse_args()


def main() -> None:
    args = _parse()

    tools = [] if args.no_tools else args.tools
    if args.tools:
        tools = args.tools

    agent = build_agent(
        model_id=args.model,
        tools=tools,
        verbosity=args.verbosity,
        max_steps=args.steps,
        plan=args.plan,
        imports=args.imports,
        stream=args.stream,
    )

    # 单次模式
    if args.task:
        result = agent.run(" ".join(args.task), return_full_result=True)
        if result.state == "max_steps_error":
            print(f"\n[!] 达到步数上限，没得出可靠结论（--steps 调大试试）", file=sys.stderr)
        print(f"\n{result.output}")
        if result.token_usage:
            u = result.token_usage
            print(
                f"[tokens in={u.input_tokens} out={u.output_tokens}"
                f" | {result.timing.duration:.1f}s]",
                file=sys.stderr,
            )
        return

    # 多轮对话模式：第一轮 reset=True，之后 reset=False 保住上下文
    print("多轮对话模式。输入 exit / Ctrl-D 退出，输入 /new 清空上下文。")
    print("提示：每轮的上下文都会累积，聊长了记得 /new 或重启。\n")

    first_turn = True
    while True:
        try:
            task = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not task:
            continue
        if task in {"exit", "quit", ":q"}:
            break
        if task == "/new":
            first_turn = True
            print("[上下文已清空]")
            continue

        try:
            answer = agent.run(task, reset=first_turn)
        except Exception as exc:  # 单轮失败不该拖垮整个会话
            print(f"[错误] {type(exc).__name__}: {exc}")
            continue

        first_turn = False
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
