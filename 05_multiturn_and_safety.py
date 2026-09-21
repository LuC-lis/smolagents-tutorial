"""
第 5 课：多轮对话 + 放权 import + 出口校验。

三个开关，都是「从 demo 走向能用」的分水岭：

    run(..., reset=False)              复用 agent.memory，不重置上下文 → 多轮对话
    additional_authorized_imports=[..] 允许模型在它写的代码里 import 别的库
    final_answer_checks=[fn]           答案出厂前的质检，不合格就让它继续改

运行：
    uv run python 05_multiturn_and_safety.py
"""

import csv
from pathlib import Path

from _shared import build_model, rule
from smolagents import CodeAgent, GoogleSearchTool

DATA = Path(__file__).resolve().parent / "demo_data"


def make_csv() -> Path:
    """造一份季度数据，让例子开箱可跑。"""
    DATA.mkdir(exist_ok=True)
    path = DATA / "quarterly.csv"
    rows = [
        ("华东", 2025, 1, 412_000),
        ("华东", 2025, 2, 388_500),
        ("华东", 2025, 3, 461_200),
        ("华东", 2025, 4, 503_400),
        ("华南", 2025, 1, 298_000),
        ("华南", 2025, 2, 356_000),
        ("华南", 2025, 3, 402_000),
        ("华南", 2025, 4, 377_500),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["region", "year", "quarter", "revenue"])
        w.writerows(rows)
    return path


CSV_PATH = make_csv()


# ===================================================================
# 1) additional_authorized_imports
# ===================================================================
# CodeAgent 会把工具调用写成 Python 代码，然后**真的执行它**。
# 所以默认只放行一小撮安全模块（见 smolagents.utils.BASE_BUILTIN_MODULES：
# collections, datetime, itertools, math, queue, random, re, stat,
# statistics, time, unicodedata）。
#
# 想让它读 CSV（需要 csv / pathlib），必须显式放权。
# 没放权时会报 InterpreterError: importing 'csv' is not allowed —— 这是设计如此，
# 不是 bug。生产环境里就别放权，或者干脆换成 docker executor。

agent = CodeAgent(
    tools=[],
    model=build_model(),
    additional_authorized_imports=["csv", "pathlib", "statistics"],
    max_steps=8,
    verbosity_level=1,
)

rule("第 1 轮：读 CSV 分析")

first = agent.run(
    f"读取 {CSV_PATH}，告诉我「华东」哪个季度的收入最高，金额是多少。",
    reset=True,
)
print("第 1 轮答案:", first)


# ===================================================================
# 2) reset=False：多轮对话，模型记得上一轮
# ===================================================================
# reset=True（默认）每次 run 都清空 memory，相当于全新会话。
# reset=False 保留 memory，模型能看到上一轮自己读过的数据和结论。
#
# 注意：memory 会越滚越长，token 花销线性增长。
# 多轮场景要自己控制：超过 N 轮就 reset，或者开一个新的 agent 做摘要压缩。

rule("第 2 轮：承接上文（reset=False）")

second = agent.run(
    "把刚才那个数字换算成万元，保留一位小数。",
    reset=False,  # ← 关键：不清空上下文
)
print("第 2 轮答案:", second)
print(f"  memory 现在有 {len(agent.memory.steps)} 步")


# ===================================================================
# 3) final_answer_checks：答案质检
# ===================================================================
# 每个 check 函数签名是 fn(final_answer, memory, agent=...) -> truthy
# 返回假值或抛异常，Agent 就会被打回去重做（并把这个错误写进上下文）。


def must_cite_url(final_answer, memory, agent=None) -> bool:
    """要求答案里必须带一个 http 链接，否则不收货。"""
    text = str(final_answer)
    if "http" not in text:
        raise ValueError("答案里没有引用来源链接，请补充链接后重新作答。")
    return True


checked_agent = CodeAgent(
    tools=[GoogleSearchTool()],
    model=build_model(),
    max_steps=6,
    verbosity_level=1,
    final_answer_checks=[must_cite_url],
)

rule("带质检的研究 Agent")

third = checked_agent.run(
    "用一句话介绍 smolagents 是什么，并附上官方文档链接。"
)
print("第 3 轮答案:", third)

# 想亲眼看看被打回是什么样？把 must_cite_url 的返回值改成 False，
# 或者让任务里不要说链接 —— 你会看到 Agent 收到质检错误后重新动手。


# ===================================================================
# 4) 把配置存成文件（部署时会用到）
# ===================================================================
# 下一课 06_export_and_reload.py 整节都在干这件事，去跑它：
#     agent.save("my_agent/")  → 生成 agent.json + prompts.yaml + app.py + tools/*.py
#     CodeAgent.from_folder("my_agent/")  → 再加载回来
# 这套目录结构正好是 agent.push_to_hub() 推上 Hugging Face Space 的格式要求。

rule("下一课")
print("  uv run python 06_export_and_reload.py")
