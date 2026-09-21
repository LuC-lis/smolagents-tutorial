"""
第 4 课：拿到实时过程 + 记账（这三个是上生产必须会的）。

    agent.run(task, stream=True)      把「内部循环」交给你，边跑边吐事件
    step_callbacks=[fn]               每一步结束回调一次，用来埋点/审计/落库
    agent.run(..., return_full_result=True)  拿到 RunResult：token 用量、耗时、完整轨迹

事件类型（从 smolagents.memory / agents 里 import）：
    PlanningStep     模型写了一段计划
    ActionStep       一步动作（含生成的代码、执行结果、错误、token 用量）
    FinalAnswerStep  最终答案
    ChatMessageStreamDelta / ToolCall / ToolOutput / ActionOutput
                     —— 只在更细粒度的流式场景出现

运行：
    uv run python 04_stream_and_hooks.py
"""

import time

from _shared import build_model, rule
from smolagents import CodeAgent
from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep, TaskStep

STEP_NAMES = {
    TaskStep: "任务",
    PlanningStep: "计划",
    ActionStep: "动作",
    FinalAnswerStep: "最终答案",
}


# ===================================================================
# 1) step_callbacks：每一步结束时被调用，签名是 fn(step, agent=...)
# ===================================================================
# 真实项目里通常用来：写日志、上报 metrics、把 token 花销记到账号上、
# 把每步存进数据库做审计。

ledger: list[dict] = []


def record_step(step, agent=None):
    usage = getattr(step, "token_usage", None)
    ledger.append(
        {
            "step": getattr(step, "step_number", None),
            "type": type(step).__name__,
            "input_tokens": usage.input_tokens if usage else 0,
            "output_tokens": usage.output_tokens if usage else 0,
            "duration": round(step.timing.duration or 0, 2),
        }
    )


agent = CodeAgent(
    tools=[],
    model=build_model(),
    max_steps=4,
    verbosity_level=0,  # 关掉自带日志，下面我们自己打印，避免两套输出打架
    step_callbacks=[record_step],
)


# ===================================================================
# 2) stream=True：run() 变成生成器，你负责迭代
# ===================================================================
rule("流式消费事件")

TASK = "把 2026 年 1 月到 12 月，每月第一天是星期几，列成一个列表。"

started = time.time()
answer = None

for event in agent.run(TASK, stream=True, reset=True):
    label = STEP_NAMES.get(type(event), type(event).__name__)
    if isinstance(event, ActionStep):
        code = (event.code_action or "").strip().splitlines()
        head = code[0] if code else "(无代码)"
        err = " ⚠ 有错误" if event.error else ""
        print(f"  → {label} #{event.step_number}: {head[:70]}{err}")
    elif isinstance(event, PlanningStep):
        print(f"  → {label}: {event.plan[:80].replace(chr(10), ' ')}…")
    elif isinstance(event, FinalAnswerStep):
        answer = event.output
        print(f"  → {label}: 到达")
    else:
        print(f"  → {label}")

print(f"\n耗时 {time.time() - started:.1f}s")
print("答案:", answer)


# ===================================================================
# 3) 记账：step_callbacks 收的数据 + RunResult
# ===================================================================
rule("成本台账（来自 step_callbacks）")

total_in = sum(r["input_tokens"] for r in ledger)
total_out = sum(r["output_tokens"] for r in ledger)
for row in ledger:
    print(
        f"  {row['type']:<16} step={str(row['step']):<5} "
        f"in={row['input_tokens']:<7} out={row['output_tokens']:<6} {row['duration']}s"
    )
print(f"  合计：输入 {total_in} tokens，输出 {total_out} tokens，共 {len(ledger)} 步")

# DeepSeek 的计价（美元 / 百万 token，示例值，按官网最新价改）：
PRICE_IN, PRICE_OUT = 0.30, 1.20
cost = total_in / 1e6 * PRICE_IN + total_out / 1e6 * PRICE_OUT
print(f"  估算费用：${cost:.6f}")


# ===================================================================
# 4) return_full_result=True：一次拿到 output + state + steps + timing
# ===================================================================
rule("RunResult")

result = agent.run(
    "3 的 20 次方除以 7 的余数是多少？只回答数字。",
    return_full_result=True,
)

print("  output      =", result.output)
print("  state       =", result.state)  # "success" 或 "max_steps_error"
print("  steps 数量  =", len(result.steps))
print("  token_usage =", result.token_usage.dict() if result.token_usage else None)
print("  duration    =", round(result.timing.duration, 2), "秒")

# state == "max_steps_error" 是必须处理的：意味着模型没做完就被掐断了。
# 这时候 result.output 是模型被逼着给出的「兜底答案」，不能当正常结果用。
