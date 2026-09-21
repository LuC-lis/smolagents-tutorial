"""
第 1 课：最小可用的 Agent。

这里用 CodeAgent —— smolagents 的招牌。它的特点是：
模型不是「调用工具」，而是**直接写一段 Python 代码**来完成任务。
上面那行 `agent.run(...)` 背后发生的事：

    for step in 1..max_steps:
        模型读上下文 -> 写一段 python 代码
        本地解释器执行这段代码（沙箱限制能 import 什么）
        把 print 出来的结果 + 返回值塞回上下文
        如果代码里调用了 final_answer(x) -> 结束，返回 x

运行：
    uv run python 01_hello.py
"""

from _shared import build_model, rule
from smolagents import CodeAgent

rule("创建 Agent")

agent = CodeAgent(
    tools=[],  # 先不给任何工具：final_answer 会被自动加上
    model=build_model(),
    # 0=静默 1=只报步骤 2=连模型的思考、代码、执行结果都打印。
    # 学习阶段强烈建议用 2，看得见才学得会。
    verbosity_level=2,
)

rule("运行 Agent")

answer = agent.run(
    "计算 1 到 20 的和，乘以 3.7，然后保留两位小数。最后用一句话告诉我结果。"
)

rule("结果")
print("答案:", answer, type(answer))


# 到这一步你已经会用了。往下走之前，先想清楚一件事：
# 上面这段代码里，没有任何一处是「你」在算数 —— 是模型自己写代码算的。
# 这就是 smolagents 的定位：让模型写代码，而不是让模型输出 JSON 让你解析。
