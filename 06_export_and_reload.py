"""
第 6 课：把 Agent 导出成文件、再加载回来（部署的起点）。

agent.save(folder) 会生成一套完整的、可独立运行的 Agent 包：

    my_agent/
    ├── agent.json          Agent 配置（注意：model 只存 model_id，不存 api_key）
    ├── prompts.yaml        完整的提示词模板，可以手改来调行为
    ├── app.py              一个能直接上 Hugging Face Space 的 Gradio 界面
    ├── requirements.txt    依赖清单
    └── tools/
        ├── reading_stats.py   你的工具被抽成了独立文件
        └── query_sales.py
    # 有 managed_agents 时还会有 managed_agents/ 目录

这正好是 agent.push_to_hub("用户名/仓库名") 需要的格式。

⚠️ 导出会被 AST 静态校验拦住，见 README §4 C 的三条规则：
    1. __init__ 参数必须有字面量默认值
    2. 工具用到的 import 写在方法内部
    3. 不要用生成器表达式
   （这三点只在导出时生效，平时 run() 完全不受影响）

运行：
    uv run python 06_export_and_reload.py
"""

import os
import shutil
from pathlib import Path

from _shared import build_model, rule
from demo_tools import DB, SalesQueryTool, reading_stats, seed_db
from smolagents import CodeAgent

EXPORT_DIR = Path(__file__).resolve().parent / "exported_agent"

seed_db()


# ===================================================================
# 1) 导出
# ===================================================================
rule("导出 Agent")
shutil.rmtree(EXPORT_DIR, ignore_errors=True)

agent = CodeAgent(
    tools=[reading_stats, SalesQueryTool(str(DB))],
    model=build_model(),
    max_steps=8,
    verbosity_level=0,
)
agent.save(EXPORT_DIR)

for path in sorted(EXPORT_DIR.rglob("*")):
    if path.is_file():
        rel = path.relative_to(EXPORT_DIR)
        print(f"  {rel}  ({path.stat().st_size} B)")


# ===================================================================
# 2) 加载回来
# ===================================================================
# 这里有个坑：agent.json 出于安全只存了 model_id，api_base 和 api_key 都不存。
# from_folder 会尝试重建模型，重建时 OpenAIModel 走的是 openai SDK 的默认逻辑，
# 也就是读环境变量 OPENAI_API_KEY / OPENAI_BASE_URL。
#
# 不设置这两个变量会直接抛：
#     openai.OpenAIError: Missing credentials.
# 即使你给 from_folder 传了 model=... 也没用 —— 它依然会先尝试重建一次。
#
# 所以：要么像下面这样把变量补上，要么干脆别用 from_folder，
# 直接重新 CodeAgent(tools=[...], model=build_model())。

rule("加载 Agent")

if os.getenv("SMOL_API_KEY"):
    os.environ.setdefault("OPENAI_API_KEY", os.environ["SMOL_API_KEY"])
    os.environ.setdefault("OPENAI_BASE_URL", os.environ.get("SMOL_API_BASE", ""))

    reloaded = CodeAgent.from_folder(EXPORT_DIR, verbosity_level=0)
    print("  加载到的工具:", list(reloaded.tools.keys()))
    print("  跑一句:", reloaded.run("用 reading_stats 统计字符串 'hello smolagents' 的词数，只回答数字"))
else:
    print("  跳过（没有 SMOL_API_KEY）")


# ===================================================================
# 3) 看看导出的工具长什么样
# ===================================================================
rule("导出的工具源码（tools/query_sales.py）")
print((EXPORT_DIR / "tools" / "query_sales.py").read_text())

print("导出的 prompts.yaml 是调行为的好地方：改文件、重载，就能改 Agent 怎么思考。")


# ===================================================================
# 4) 上 Hugging Face Space
# ===================================================================
# 需要 hf 登录（uv run hf auth login）：
#
#     agent.push_to_hub("你的用户名/my-smol-agent")
#
# 它会建一个 gradio space 并上传，app.py 已经生成好了，打开网页就能用。
