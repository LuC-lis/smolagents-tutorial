# smolagents 上手笔记

一个能跑的中文教程 + 示例集。每一步都在本机实测跑通过。

- Python 3.12 / `smolagents 1.26.0` / 用 `uv` 管理
- 模型：**DeepSeek**（走 OpenAI 兼容协议），已配在 `.env` 里
- 想换模型只看 [§5 换模型](#5-换模型)，一行环境变量的事

---

## 0. 三十秒上手

```bash
uv run smol "2 的 100 次方有多少位数字？"     # 单次跑一个任务
uv run smol                                   # 进入多轮对话模式（Ctrl-D 退出）
```

就这两条。下面全是「为什么」。

---

## 1. 先认清两个概念

整本书就这两个词重要：

| | 是什么 | 你会换它的时候 |
|---|---|---|
| **Model** | 只会「一轮对话」的脑子。给它一串 messages，它回一段文字或一个工具调用。它不知道什么叫循环，也不会执行代码。 | 换供应商、换模型、调温度 |
| **Agent** | 用这个脑子的**流程**。它负责：循环、选工具、解析并真的执行 Python 代码、把结果塞回上下文、直到给出 `final_answer`。 | 改工作方式、加工具、加约束 |

所以：**换模型永远只改 Model，改行为永远只改 Agent。** 两者在代码里是分开传的：

```python
agent = CodeAgent(
    model=build_model(),   # ← Model
    tools=[...],           # ← Agent 的配置
)
```

### CodeAgent vs ToolCallingAgent

`CodeAgent` 是 smolagents 的招牌，也是本教程默认用的：

| | CodeAgent | ToolCallingAgent |
|---|---|---|
| 模型怎么「动手」 | **写一段 Python 代码**，本地执行 | 输出结构化 JSON 工具调用 |
| 能力上限 | 能组合、循环、条件判断、调 `math`/`re` | 一次只能调一个预定义工具 |
| 谁来执行 | 真的执行模型写的代码 | 不执行代码 |
| 风险 | 需要沙箱约束（见 §7） | 更安全，但天花板低 |
| 适合 | 数据处理、多步推理、复杂流程 | 只做「问答 → 调接口」的简单场景 |

想换成后者只需改一个类名，`tools`/`model` 参数完全一样：

```python
from smolagents import ToolCallingAgent
agent = ToolCallingAgent(tools=[...], model=build_model())
```

> 官方原话：CodeAgent 在相同任务上通常用更少的步骤、更少的 token 拿到更好的结果。代价是你必须信任沙箱。

---

## 2. 逐课示例（按顺序跑）

```bash
uv run python 01_hello.py
uv run python 02_custom_tool.py
uv run python 03_web_research.py
uv run python 04_stream_and_hooks.py
uv run python 05_multiturn_and_safety.py
uv run python 06_export_and_reload.py
```

| 文件 | 讲什么 | 跑完你应该记住 |
|---|---|---|
| `01_hello.py` | 最小可用 Agent，`verbosity_level=2` | Agent 的循环长什么样：模型写代码 → 本地执行 → 结果回灌 → `final_answer()` 收尾 |
| `02_custom_tool.py` + `demo_tools.py` | `@tool` 装饰器 / 继承 `Tool` 类 | 工具的 `description` 就是给模型的「岗位说明书」；`forward` 签名必须和 `inputs` 对齐 |
| `03_web_research.py` | 内置联网工具 + `planning_interval` | 工具不是重点，**怎么组织流程**才是。多步任务一定要开 planning + `max_steps` |
| `04_stream_and_hooks.py` | `stream=True`、`step_callbacks`、`RunResult` | 怎么拿到实时进度、怎么记 token 账、怎么判断「跑完了」还是「被掐断了」 |
| `05_multiturn_and_safety.py` | `reset=False`、`additional_authorized_imports`、`final_answer_checks` | 多轮记忆、放权 import、答案出厂质检 |
| `06_export_and_reload.py` | `agent.save()` / `from_folder()` | 怎么把 Agent 变成一个可独立部署的包（HF Space 的格式） |

每个文件都写了大段注释，直接读注释比读这份 README 有用。

---

## 3. Agent 的执行循环（看懂这个，就不需要看文档了）

```
          ┌──────────────────────────────────────────┐
          │  系统提示词（含所有工具的 name/description/inputs）│
          └──────────────────┬───────────────────────┘
                             ▼
   ┌─────────────────────────────────────────────────────┐
   │ 1. 把 memory 里所有历史 step 拼成 messages 发给模型    │
   │ 2. 模型回一段 Python 代码（用 <code> 或 ```python 包裹）│
   │ 3. 解析这段代码                                       │
   │ 4. 丢进 LocalPythonExecutor 执行                      │
   │      ├─ 有 final_answer(x)  → 结束，返回 x             │
   │      ├─ 抛异常             → 错误信息回灌，回到 1       │
   │      └─ 正常返回           → 打印结果回灌，回到 1       │
   └─────────────────────────────────────────────────────┘
                        （超过 max_steps 强制收尾）
```

`agent.memory.steps` 就是上面这个循环的完整流水账，类型有四种：

- `TaskStep` —— 你给的任务
- `PlanningStep` —— 模型写的计划（开了 `planning_interval` 才有）
- `ActionStep` —— 一步动作，包含 `.code_action`（生成的代码）、`.observations`、`.error`、`.token_usage`、`.timing`
- `FinalAnswerStep` —— 最终答案

```python
for i, step in enumerate(agent.memory.steps):
    print(i, type(step).__name__)
```

---

## 4. 工具：三种写法

### A. `@tool` 装饰器（最常用）

```python
from smolagents import tool

@tool
def reading_stats(text: str) -> str:
    """统计一段文本的字符数、词数和最长的词。

    Args:
        text: 要统计的文本内容。
    """
    words = text.split()
    return f"字符数 {len(text)}，词数 {len(words)}"
```

三条硬性要求，缺一个就报错：

1. **必须有返回类型注解**（没写会抛 `TypeHintParsingException`）
2. **docstring 必须有 `Args:` 段**（这是模型唯一的参数说明来源）
3. **函数必须写在真正的 `.py` 文件里** —— smolagents 会 `inspect.getsource()` 读源码写进 prompt，
   在 `python -c "..."` 或 REPL 里定义会报 `OSError: could not get source code`

### B. 继承 `Tool` 类（需要 `__init__` 或非文本输出时）

```python
class SalesQueryTool(Tool):
    name = "query_sales"                      # 模型代码里调用的函数名
    description = "查询本地销售数据库，输入一条 SELECT 语句..."   # 模型据此选工具
    inputs = {"sql": {"type": "string", "description": "一条 SELECT 语句"}}
    output_type = "string"

    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path

    def forward(self, sql: str) -> str:       # 签名必须和 inputs 对得上
        ...
```

`inputs` 里的 `type` 只能是：`string` `boolean` `integer` `number` `image` `audio` `array` `object` `any` `null`。

### C. 想让 `agent.save()` 成功导出工具？三条额外规则

工具要能被序列化成独立的 `.py` 文件（这样才能 `from_folder` 加载、推到 HF Space），
smolagents 会用 AST 静态校验**这个类自己的源码**。实测会拦下这些写法：

```python
# ❌ __init__ 的参数必须有「字面量」默认值
#    报：Parameters in __init__ must have default values, found required parameters: db_path
#    报：Parameters in __init__ must have literal default values, found non-literal defaults: db_path
def __init__(self, db_path: Path = Path("x.db")): ...

# ✅ 用 str / int / None / [] / {} 这类字面量
def __init__(self, db_path: str = "x.db"): ...
```

```python
# ❌ 模块顶部的 import 不在类源码里，校验器看不见
#    报：- forward: Name 'sqlite3' is undefined
import sqlite3

class MyTool(Tool):
    def forward(self, x: str) -> str:
        return sqlite3.connect(...)   # ← 校验失败

# ✅ 把 import 写进方法内部
class MyTool(Tool):
    def forward(self, x: str) -> str:
        import sqlite3
        return sqlite3.connect(...)
```

```python
# ❌ 校验器认识列表推导，不认识生成器表达式
#    报：- forward: Name 'cell' is undefined
return "\n".join(" | ".join(str(c) for c in row) for row in rows)

# ✅ 用列表推导或普通 for 循环
lines = []
for row in rows:
    lines.append(" | ".join([str(cell) for cell in row]))
return "\n".join(lines)
```

这三条只在 `agent.save()` / `push_to_hub()` 时触发，平时 `run()` 完全不受影响。
所以开发时怎么顺手怎么写，要导出了再来改。`02_custom_tool.py` 里的 `SalesQueryTool`
已经按这套规矩写好了。

关于「工具源码长什么样」，可以直接看导出的结果：

```python
agent.save("my_agent/")
# my_agent/
# ├── agent.json           Agent 的配置（注意：model 只存 model_id，api_key 不存）
# ├── prompts.yaml         完整的提示词模板
# ├── app.py               可以直接上 HF Space 的 Gradio 界面
# ├── requirements.txt     依赖清单
# └── tools/
#     ├── reading_stats.py
#     └── query_sales.py   ← 就是你的 Tool 类，加上自动补的 import
```

### D. 随时脱离 Agent 单独测试工具

```python
print(reading_stats("hello world"))     # 工具本身就是可调用对象
```

**写工具时永远先这样自测**，别一上来就跑 Agent，不然你分不清是工具坏了还是模型选错了。

### 本机内置工具可用性（实测）

| 工具 | 状态 | 说明 |
|---|---|---|
| `PythonInterpreterTool` | ✅ | 名字是 `python_interpreter` |
| `VisitWebpageTool` | ✅ | 抓网页转 markdown，需要 `markdownify`（已装） |
| `WikipediaSearchTool` | ✅ | 需要 `wikipedia-api`（已装） |
| `GoogleSearchTool` | ✅ | 需要 `SERPAPI_API_KEY`（本机已配），工具名是 `web_search` |
| `DuckDuckGoSearchTool` | ❌ | 本机访问不通（连 yahoo 超时） |
| `WebSearchTool` | ❌ | 依赖 DuckDuckGo，同上 |
| `SpeechToTextTool` | ❌ | 需要 `smolagents[audio]` |

`add_base_tools=True` 会一次性挂上 `DuckDuckGoSearchTool + VisitWebpageTool + PythonInterpreterTool`，
但因为 DuckDuckGo 在本机不通，这里用 `GoogleSearchTool` 替代：

```python
CodeAgent(tools=[GoogleSearchTool(), WikipediaSearchTool(), VisitWebpageTool()], ...)
```

### 接入 MCP 服务器

```python
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters

with ToolCollection.from_mcp(
    {"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"}
) as tc:
    agent = CodeAgent(tools=[*tc.tools], model=build_model())
    agent.run("...")
```

需要 `uv add "smolagents[mcp]"`。本地 stdio 服务器就传 `StdioServerParameters(command=..., args=[...])`。

---

## 5. 换模型

只改 `.env` 里三行，或者用 `--model` 临时覆盖。

| 供应商 | `SMOL_API_BASE` | `SMOL_MODEL` 例子 |
|---|---|---|
| DeepSeek（当前） | `https://api.deepseek.com/v1` | `deepseek-flash` / `deepseek-v4-pro` |
| OpenAI | `https://api.openai.com/v1` | `gpt-5` |
| OpenRouter | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-4.5` |
| Ollama 本地 | `http://localhost:11434/v1` | `qwen3:8b`（key 随便填） |
| vLLM / one-api / LiteLLM 网关 | 你的网关地址 | 你的模型名 |

原理：`OpenAIModel` 只是把 `api_base` 丢给 `openai.OpenAI(base_url=...)`，所以**任何实现了 `/chat/completions` 的服务都能接**。

```python
OpenAIModel(model_id="deepseek-flash", api_base="https://api.deepseek.com/v1", api_key="...")
```

其他原生 client（一般用不上）：

```python
from smolagents import InferenceClientModel, LiteLLMModel, TransformersModel, MLXModel
InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Thinking")   # HF，走 HF_TOKEN
LiteLLMModel(model_id="anthropic/claude-sonnet-4.5")                # 100+ 供应商，需 smolagents[litellm]
TransformersModel(model_id="Qwen/Qwen2.5-Coder-7B-Instruct")        # 本地跑，需 smolagents[torch]
```

> 小模型（&lt;7B）经常写不对 CodeAgent 要求的代码格式。CodeAgent 对模型能力有要求，
> 优先选「会写代码」的模型。

---

## 6. 参数速查

### `CodeAgent(...)`

| 参数 | 默认 | 什么时候动它 |
|---|---|---|
| `tools` | — | 必填（可以是 `[]`） |
| `model` | — | 必填 |
| `max_steps` | `20` | **一定要调小**，10 左右够大多数任务。这是防跑飞的安全绳 |
| `verbosity_level` | `1` | 学习用 `2`（打印代码+执行结果），生产用 `0`/`1` |
| `planning_interval` | `None` | 任务超过 5 步就设成 `3` |
| `additional_authorized_imports` | `[]` | 需要 import 白名单以外的模块时 |
| `final_answer_checks` | `[]` | 答案有硬性格式要求时 |
| `step_callbacks` | `None` | 要埋点、记账、写审计日志时 |
| `return_full_result` | `False` | 需要知道 `state`/`token_usage`/耗时 |
| `stream_outputs` | `False` | 想在终端看见字一个个冒出来 |
| `executor_type` | `"local"` | 换 `docker`/`e2b`/`modal`/`blaxel` 做真沙箱 |
| `instructions` | `None` | 附加系统提示，用来定人设/加约束 |
| `managed_agents` | `None` | 子 Agent 团队（见 §9） |

### `agent.run(...)`

| 参数 | 默认 | 说明 |
|---|---|---|
| `task` | — | 任务描述。**写得越具体，结果越稳** |
| `reset` | `True` | `False` = 保留上文，做多轮对话 |
| `stream` | `False` | `True` 时返回生成器，要自己 `for` 迭代 |
| `max_steps` | `None` | 单次覆盖 agent 的 `max_steps` |
| `return_full_result` | `None` | `True` 返回 `RunResult` |
| `additional_args` | `None` | 往执行器里注入变量，如 `{"df": my_dataframe}` |

### `RunResult`

```python
result = agent.run(task, return_full_result=True)
result.output       # 最终答案（AgentText / AgentImage / ...）
result.state        # "success" | "max_steps_error"   ← 必须判断这个
result.steps        # 完整轨迹
result.token_usage  # .input_tokens / .output_tokens / .total_tokens
result.timing       # .start_time / .end_time / .duration
```

`result.state == "max_steps_error"` 意味着模型**没做完就被掐断了**，
这时 `result.output` 是被逼出来的兜底答案，不能当正常结果用。

### 返回值类型

`agent.run()` 返回的不是裸 `str`，是 `AgentText`。它继承 `str`，能当字符串用；
要拿原始值用 `.to_raw()`。还有 `AgentImage` / `AgentAudio`。

---

## 7. 常见坑（都是实测踩过的）

**1. `@tool` 报 `OSError: could not get source code`**
装饰器要读函数源码。把工具函数放进 `.py` 文件，别在 REPL / `-c` 里定义。

**2. `TypeHintParsingException: Tool return type not found`**
忘了写返回类型注解。`def f(x: str) -> str:` ← 箭头后面那个必须有。

**3. `InterpreterError: importing 'csv' is not allowed`**
这是**设计如此**，不是 bug。CodeAgent 默认只放行
`collections, datetime, itertools, math, queue, random, re, stat, statistics, time, unicodedata`。
要读 CSV 就 `additional_authorized_imports=["csv", "pathlib"]`。

**4. 工具实例化时报 `AttributeError` / `TypeError: You must set an attribute ...`**
继承 `Tool` 时 `name`/`description`/`inputs`/`output_type` 四个都得写，且类型要对。
另外 `forward` 的参数名和类型必须和 `inputs` 完全一致。

**5. 模型不用我给的参数名调工具**
它只看得见 `inputs` 里的 key。名字要起得直白（`sql`、`text`），别用 `q`、`s`。

**6. Agent 绕圈烧 token**
`max_steps=10` 卡死；任务描述写清楚「要什么、什么格式、什么算完成」；
多步任务开 `planning_interval=3`。

**7. 用 `verbosity_level=2` 时输出被 rich 折行折得乱七八糟**
终端窄。调宽窗口，或者用 `verbosity_level=1`。

**8. `stream=True` 之后 `run()` 不干活了**
它是生成器，不迭代就不执行。`for event in agent.run(..., stream=True): ...`

**9. 多轮对话 token 越滚越多**
`reset=False` 时 memory 只增不减。超 N 轮就 `reset=True` 重开，或者开个新 Agent 让模型压缩摘要。

**10. DuckDuckGo 搜索超时**
本机网络到 ddgs 不通。用 `GoogleSearchTool`（本机已配 `SERPAPI_API_KEY`）。
`add_base_tools=True` 挂的是 DuckDuckGo，本机别用。

**11. 这个仓库的 `.gitignore` 内容是 `*`**
意味着 `git add .` 什么都不加。要提交得先把 `.gitignore` 改掉，或者 `git add -f <文件>`。
（大概率是之前误建的）

**12. 工具用生成器表达式 → `agent.save()` 报 `Name 'cell' is undefined`**
见 §4 C。校验器认识列表推导、`for` 循环，就是不认识生成器表达式。

**13. `__init__` 的参数没默认值 → `agent.save()` 报 `must have default values`**
而且默认值必须是**字面量**，`Path("x")` 这种也不行，得写 `db_path: str = "x.db"`。

**14. `CodeAgent.from_folder()` 报 `openai.OpenAIError: Missing credentials`**
因为 `agent.json` 出于安全**只存 `model_id`，不存 api_base 和 api_key**。
`from_folder` 会尝试重建模型，重建时读的是环境变量，所以要么

```bash
export OPENAI_API_KEY=你的key
export OPENAI_BASE_URL=https://api.deepseek.com/v1
```

……要么别用 `from_folder`，直接重新 `CodeAgent(tools=[...], model=build_model())`。
注意：即使你传了 `model=...` 给 `from_folder`，它依然会先试着重建一次保存的模型，
所以这个错还是躲不掉。

**15. `result.output` 不是裸 `str`**
是 `AgentText`（继承 `str`，能用），要原始值用 `.to_raw()`。
也有 `AgentImage` / `AgentAudio`。

---

## 8. 安全：CodeAgent 会真的执行模型写的代码

默认执行器是 `LocalPythonExecutor`，它是一个**受限解释器**（不是 CPython 子进程），
有这些限制：

- import 白名单（见坑 #3）
- 禁 dunder 属性访问（`__class__`、`__globals__` 等）
- `MAX_OPERATIONS = 10_000_000` 次操作上限
- `MAX_EXECUTION_TIME_SECONDS = 30` 秒
- `MAX_WHILE_ITERATIONS = 1_000_000`

**它够安全吗？不够。** 它挡的是「模型手滑」，不是「模型被 prompt 注入后主动搞破坏」。
只要你的工具能读文件、能发网络请求，被注入的模型就能用你的权限去读、去发。

生产环境的做法：

```python
CodeAgent(..., executor_type="docker")   # 或 "e2b" / "modal" / "blaxel"
```

需要 `uv add "smolagents[docker]"` 等对应 extra。远程执行器 = 真沙箱，
模型写的代码在隔离容器里跑，你的 `.env` 和文件系统它碰不到。

另外**永远不要把 `additional_authorized_imports=["*"]`** 放进生产代码。

---

## 9. 下一步可以玩什么

**子 Agent 团队（managed agents）** —— 主 Agent 可以把子任务委派给专职 Agent：

```python
researcher = CodeAgent(
    tools=[GoogleSearchTool()],
    model=build_model(),
    name="researcher",
    description="负责联网查资料并返回结论和来源链接",
)
manager = CodeAgent(
    tools=[],
    model=build_model(),
    managed_agents=[researcher],   # 主 Agent 就能像调工具一样调用它
)
manager.run("调研 xxx，给我一份带来源的结论")
```

**把 Agent 存成文件 / 推上 HF Space**：

```python
agent.save("my_agent/")
# 生成：agent.json + prompts.yaml + app.py + requirements.txt + tools/*.py
```

完整可跑的版本在 `06_export_and_reload.py`，包括导出、加载、打印生成的工具源码。

⚠️ 导出前先满足 §4 C 的三条规则，否则 `save()` 会直接用 AST 校验把你拦下来
（实测：生成器表达式、必需的 `__init__` 参数、模块顶部的 import 都会报错）。

加载回来的坑：`agent.json` **不存 api_key / api_base**，`from_folder` 重建模型时读的是环境变量，
实测可行的写法：

```bash
export OPENAI_API_KEY=$SMOL_API_KEY
export OPENAI_BASE_URL=https://api.deepseek.com/v1
uv run python 06_export_and_reload.py
```

**做个网页界面**：`uv add "smolagents[gradio]"` 然后 `GradioUI(agent).launch()`

**官方 CLI**（本机自带，参数没预设所以要手动传）：

```bash
uv run smolagent --model-type OpenAIModel --api-base https://api.deepseek.com/v1 \
    --api-key $SMOL_API_KEY --model-id deepseek-flash "你的任务"
```

---

## 10. 文件地图

```
smol/
├── .env                    你的密钥（已配好 DeepSeek）★ 不要提交
├── .env.example            模板
├── _shared.py              build_model()：所有示例共用的 Model 工厂
├── demo_tools.py           示例工具（reading_stats + SalesQueryTool）
├── 01_hello.py             最小可用 Agent
├── 02_custom_tool.py       两种自定义工具写法
├── 03_web_research.py      内置联网工具 + planning
├── 04_stream_and_hooks.py  流式事件 / 回调记账 / RunResult
├── 05_multiturn_and_safety.py  多轮 / 放权 import / 答案质检
├── 06_export_and_reload.py     导出成可部署的 Agent 包，再加载回来
├── src/smol/__init__.py    `uv run smol` 的 CLI 实现（多轮对话模式）
├── exported_agent/         06 跑完自动生成，可删
└── demo_sales.db, demo_data/   示例自动生成的假数据，可删
```

自己改东西：

```bash
uv add <包>              # 装依赖（不要用 pip，会破坏 uv.lock）
uv run python 01_hello.py
```

---

## 附：官方资料

- 文档 https://huggingface.co/docs/smolagents
- 仓库 https://github.com/huggingface/smolagents
- 安装好的源码就在本地，比文档准：
  ```bash
  uv run python -c "import smolagents, os; print(os.path.dirname(smolagents.__file__))"
  ```
  重点看 `agents.py`（`MultiStepAgent.run` 的主循环）、`tools.py`（`Tool` / `@tool`）、
  `local_python_executor.py`（沙箱白名单）、`default_tools.py`（内置工具照抄模板）。
