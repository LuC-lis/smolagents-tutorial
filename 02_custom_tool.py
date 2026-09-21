"""
第 2 课：给 Agent 加工具。

模型本身只会写字。工具就是「模型的手」——真正去查数据库、调 API、
读文件的那部分。

工具 = 三样东西：
    name         模型在代码里调用它时用的函数名
    description  模型判断「什么时候该用它」的唯一依据（写不好就选错工具）
    inputs       每个参数的 JSON schema，会原样进 prompt

两种写法，按需选：
    A. @tool 装饰器   —— 函数 + 类型注解 + docstring，最省事
    B. 继承 Tool 类   —— 显式声明上面三样，能做初始化、能返回图片/音频

两个工具本身定义在同目录的 demo_tools.py 里（真实项目也该这么放），
本文件只负责讲解 + 演示怎么用。建议对照着 demo_tools.py 一起看。

运行：
    uv run python 02_custom_tool.py
"""

from _shared import build_model, rule
from demo_tools import DB, SalesQueryTool, reading_stats, seed_db
from smolagents import CodeAgent

seed_db()

# ===================================================================
# 先单独测工具，别一上来就跑 Agent
# ===================================================================
# 工具本身就是可调用对象。不先自测就跑 Agent，你会分不清是
# 「工具坏了」还是「模型选错了工具」—— 这是最浪费时间的一类 bug。

rule("单独测试 @tool 风格的工具")
print(reading_stats("smolagents lets models write Python code"))
print("  自动推导出的 inputs =", reading_stats.inputs)
# → inputs 就是 docstring 里的 Args 段 + 类型注解，一起生成的。

rule("单独测试 class 风格的工具")
sales_tool = SalesQueryTool(str(DB))
print(sales_tool("SELECT region, SUM(amount) FROM sales GROUP BY region ORDER BY 2 DESC"))
print("  拒绝非 SELECT ->", sales_tool("DROP TABLE sales"))
# → 工具里可以写自己的防御逻辑。模型写的代码不可全信，边界的校验放在工具里。


# ===================================================================
# 把工具交给 Agent
# ===================================================================
rule("Agent 自主选择工具")

agent = CodeAgent(
    tools=[reading_stats, sales_tool],
    model=build_model(),
    verbosity_level=2,
    max_steps=6,
)

agent.run(
    "哪个区域的销售额最高？给出它的总销售额，并顺手告诉我「华南」这两个字的字符数。"
)

# 观察点：看它第 1 步往哪个工具上撞。如果在两个工具之间犹豫，
# 说明 description 写得不够清楚 —— 工具的 description 就是它的「岗位说明书」。

# ------------------------------------------------------------------
# 顺手记两条 @tool 的硬性要求（都实测踩过）：
#
#   1. 必须有返回类型注解。`def f(x: str) -> str:` 箭头后面那个必须有，
#      否则抛 TypeHintParsingException: Tool return type not found。
#
#   2. docstring 必须有 Args: 段，且函数必须写在真正的 .py 文件里。
#      smolagents 会 inspect.getsource() 读源码写进 prompt，
#      在 python -c "..." 或 REPL 里定义会报 OSError: could not get source code。
#
# 还有一条只在 agent.save() 时才会撞上的规矩（见 README §4 C 和 06）：
#   class 风格的工具里，__init__ 参数要有字面量默认值、import 要写在方法内部、
#   别用生成器表达式。demo_tools.py 里的注释标了具体位置。
# ------------------------------------------------------------------
