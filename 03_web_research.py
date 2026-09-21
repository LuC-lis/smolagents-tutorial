"""
第 3 课：内置工具 + 规划（planning）。

到这里你会发现一件事：**工具不是重点，怎么组织流程才是。**
第 3 课引入 smolagents 的另一个关键开关 —— planning_interval。

    planning_interval=3
    意思是：每跑 3 步动作，就插一次「先别动手，先写下计划」。
    模型会输出一个 numbered plan，存进 memory，后续步骤都带着它。

单步任务用不上；一旦任务需要 5 步以上（查资料 → 打开链接 → 交叉验证 → 汇总），
planning 能明显减少跑偏和绕圈。

内置工具说明（本机实测）：
    GoogleSearchTool    需要 SERPAPI_API_KEY，本机已配好，可用 ✅
    VisitWebpageTool    抓网页转 markdown，可用 ✅
    WikipediaSearchTool 可用 ✅
    DuckDuckGoSearchTool 本机访问不通（超时），所以这里不用它 ❌
    想一次性全塞进去就用 add_base_tools=True（但它挂的是 DuckDuckGo，本机会超时）

运行：
    uv run python 03_web_research.py
"""

from _shared import build_model, rule
from smolagents import CodeAgent, GoogleSearchTool, VisitWebpageTool, WikipediaSearchTool

rule("组装工具")

search = GoogleSearchTool()  # 工具名就是 web_search
wiki = WikipediaSearchTool()
visit = VisitWebpageTool(max_output_length=8000)  # 单页最多喂给模型 8000 字符

rule("创建带规划的研究 Agent")

agent = CodeAgent(
    tools=[search, wiki, visit],
    model=build_model(),
    planning_interval=3,  # 每 3 步重新规划一次
    max_steps=10,  # 硬上限，防止无限绕圈烧 token
    # 这一步输出会很长，调到 1 —— 只报步骤和结论，不打印完整代码。
    # 想看清细节就改回 2。
    verbosity_level=1,
)

rule("开始研究")

answer = agent.run(
    "smolagents 1.26 里 CodeAgent 和 ToolCallingAgent 的区别是什么？"
    "请用官方文档核实，然后给我一段不超过 150 字的对比结论，并给出你引用的链接。"
)

rule("结论")
print(answer)


# 事后复盘：agent.memory.steps 里存着完整轨迹。
# 打印每步的类型，看得见规划步骤在哪里插入：
rule("轨迹回放")
for i, step in enumerate(agent.memory.steps):
    print(f"  [{i:02d}] {type(step).__name__}")
