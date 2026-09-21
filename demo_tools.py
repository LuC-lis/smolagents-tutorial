"""
两个示例工具，被 02 和 06 共用。

真实项目里的目录结构大概也是这样：工具单独放一个模块，脚本 import 它。
（因为 smolagents 要读工具的源码，工具必须定义在真正的 .py 文件里。）
"""

import sqlite3
from pathlib import Path

from smolagents import Tool, tool

DB = Path(__file__).resolve().parent / "demo_sales.db"


# ===================================================================
# A 类工具：@tool 装饰器 —— 最省事，适合普通函数
# ===================================================================

@tool
def reading_stats(text: str) -> str:
    """统计一段文本的字符数、词数和最长的词。

    Args:
        text: 要统计的文本内容。
    """
    words = text.split()
    chars = len(text)
    longest = max(words, key=len) if words else "(空)"
    return f"字符数 {chars}，词数 {len(words)}，最长的词是「{longest}」"


# ===================================================================
# B 类工具：继承 Tool —— 需要 __init__ 传配置、或要返回图片/音频时用
# ===================================================================

class SalesQueryTool(Tool):
    """在本地 SQLite 上执行只读 SELECT。"""

    name = "query_sales"
    description = (
        "查询本地销售数据库。库中有一张表 sales(region TEXT, month TEXT, amount REAL)。"
        "输入一条 SELECT 语句，返回结果行（最多 20 行）。只允许只读查询。"
    )
    inputs = {
        "sql": {
            "type": "string",
            "description": "一条 SQLite 的 SELECT 语句，例如：SELECT region, SUM(amount) FROM sales GROUP BY region",
        }
    }
    output_type = "string"

    # ⚠️ 下面这些写法不是「优雅」，是 agent.save() 导出工具时的硬性要求。
    #    想把 Agent 存成文件、或推到 Hugging Face Space，就得守这两条规矩：
    #
    #    1) __init__ 的参数必须有「字面量」默认值（"x" / 1 / None / [] / {}）。
    #       写 db_path: Path = Path("x") 会报 "must have literal default values"，
    #       所以这里用 str 而不是 Path。
    #
    #    2) 工具用到的 import 要写在**方法内部**。导出时校验器只看这个类的源码，
    #       模块顶部的 `import sqlite3` 它看不见，会报 "Name 'sqlite3' is undefined"。
    #
    #    3) 顺带一提：类里别用生成器表达式。校验器认识列表推导 [x for x in y]，
    #       却不认识生成器 (x for x in y)，会误报 "Name 'cell' is undefined"。
    def __init__(self, db_path: str = "demo_sales.db"):
        super().__init__()
        self.db_path = db_path

    def forward(self, sql: str) -> str:
        # 这个签名必须和上面的 inputs 完全对得上，
        # 否则实例化时 validate_arguments() 会直接抛异常。
        import sqlite3  # 见上面第 2 条

        if not sql.strip().lower().startswith("select"):
            return "拒绝执行：只允许 SELECT 查询。"

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql).fetchmany(20)

        if not rows:
            return "（没有匹配的数据）"

        lines = []
        for row in rows:
            lines.append(" | ".join([str(cell) for cell in row]))
        return "\n".join(lines)


# ===================================================================
# 造点假数据，让例子开箱可跑
# ===================================================================

def seed_db() -> None:
    data = [
        ("华东", "2026-01", 128000.0),
        ("华东", "2026-02", 141500.0),
        ("华东", "2026-03", 99000.0),
        ("华北", "2026-01", 87000.0),
        ("华北", "2026-02", 92300.0),
        ("华北", "2026-03", 110400.0),
        ("华南", "2026-01", 156000.0),
        ("华南", "2026-02", 132000.0),
        ("华南", "2026-03", 168500.0),
    ]
    with sqlite3.connect(DB) as conn:
        conn.execute("DROP TABLE IF EXISTS sales")
        conn.execute("CREATE TABLE sales(region TEXT, month TEXT, amount REAL)")
        conn.executemany("INSERT INTO sales VALUES (?, ?, ?)", data)
