# jobagent

`jobagent` 是一个本地优先的 AI 面试备战计划生成器。输入目标公司、岗位和 JD 后，它会结合联网岗位调研、本地八股/面经知识库检索、reranker 重排和 LLM 结构化生成，输出一份带引用来源的面试准备计划。

这个项目当前定位是可本地演示、可写进简历、可在面试中讲清楚工程链路的 MVP，不是生产级 SaaS。

## 当前能力

- CLI 和 Streamlit UI 两个入口，复用同一套 LangGraph workflow。
- Tavily 联网搜索公司、岗位、面试流程和面经线索。
- Chroma 本地向量库检索 `data/baguwen/` 下的自写样例知识库。
- BGE embedding + CrossEncoder reranker 对本地 KB 候选重排。
- DeepSeek OpenAI-compatible API 生成结构化 JSON 备战计划。
- Pydantic schema 校验、`source_id` 存在性校验、引用回填和 validate -> generate 重试。
- 失败路径可见：CLI / UI 会展示 Tavily、KB、LLM 等 warning，而不是静默吞掉。
- 每次运行保存 `evidence.json`、`plan.json`、`plan.md`，方便复现 demo。

## 工作流

```text
JobInput(company, role, jd_text)
  |
  v
intake
  - 生成 web search queries
  - 生成本地知识库 kb_query
  |
  v
research
  - Tavily web search
  - 去重并限制 web evidence 数量
  |
  v
retrieve
  - Chroma 向量召回本地 KB 候选
  |
  v
rerank
  - CrossEncoder reranker 重排 KB 候选
  - 统一分配稳定证据 id: web-N / kb-N
  |
  v
generate
  - DeepSeek 生成结构化 JSON
  |
  v
validate
  - schema 校验
  - source_id 校验
  - citation 回填
  - 失败时回到 generate 重试
  |
  v
render
  - 写入 evidence.json / plan.json / plan.md
```

## 快速开始

### 1. 安装依赖

项目使用 Python 3.11 和 `uv`：

```bash
uv sync
```

### 2. 配置环境变量

复制示例配置：

```bash
cp .env.example .env
```

至少需要填写：

```bash
DEEPSEEK_API_KEY=你的 DeepSeek API key
```

可选配置：

```bash
TAVILY_API_KEY=你的 Tavily API key
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=jobagent
```

说明：

- `DEEPSEEK_API_KEY` 是生成备战计划的必需配置。
- `TAVILY_API_KEY` 缺失时，联网调研会产生 warning；workflow 仍会尝试基于本地 KB 和通用建议继续。
- `EMBEDDING_MODEL_PATH` / `RERANKER_MODEL_PATH` 可指向本地模型路径；未设置时使用 `.env.example` 中的默认模型名。

### 3. 构建本地知识库索引

```bash
uv run python scripts/build_index.py --data-dir data/baguwen --persist-dir .chroma/jobagent
```

当前样例语料会索引为 162 个 chunk。Chroma 可能打印类似 `capture() takes 1 positional argument but 3 were given` 的 telemetry warning；当前验证中索引构建和查询不受影响。

### 4. 运行 CLI

使用模块入口：

```bash
uv run python -m src.cli plan \
  --company "阿里巴巴" \
  --role "Agent 工程师" \
  --jd-file examples/jd_agent_engineer.txt \
  --out runs/demo
```

也可以使用 `pyproject.toml` 中注册的脚本入口：

```bash
uv run jobagent plan \
  --company "阿里巴巴" \
  --role "Agent 工程师" \
  --jd-file examples/jd_agent_engineer.txt \
  --out runs/demo
```

CLI 运行时会显示各节点进度、web / KB 证据计数和 warning。

### 5. 运行 Streamlit UI

```bash
uv run streamlit run app.py
```

UI 会展示输入框、生成结果、证据计数、warning，以及 `plan.md` / `plan.json` 下载按钮。

## 输出文件

默认输出目录由 `--out` 或 UI 的输出目录字段控制，例如 `runs/demo/`：

```text
runs/demo/
├── evidence.json
├── plan.json
└── plan.md
```

- `evidence.json`：最终提供给 LLM 的 web / KB 证据。
- `plan.json`：LLM 结构化输出和校验后的计划数据。
- `plan.md`：面向阅读和 demo 展示的 Markdown 备战计划。

`runs/` 已加入 `.gitignore`，默认不提交生成结果。

## 检索评测

当前内置固定 eval 集：

- 文件：`src/eval/eval_set.json`
- 覆盖：RAG、Agent、数据库、网络、并发、系统设计、机器学习指标、Linux 排障等 12 个问题。
- gold 标注：使用 `source + heading`，避免依赖 Chroma chunk id。

运行方式：

```bash
uv run python -m src.eval.retrieval_eval
```

2026-06-14 基线结果：

| mode | hit@3 | hit@5 | MRR |
| --- | ---: | ---: | ---: |
| no_rerank | 1.000 | 1.000 | 1.000 |
| rerank | 1.000 | 1.000 | 1.000 |

解释：

- 当前 12 题在纯向量召回下已经全部 top-1 命中，因此 reranker 没有体现可见指标提升。
- 这说明当前样例语料和 eval set 偏容易；后续需要增加更难的换述题、干扰主题和参数扫描，才能更可靠地分析 reranker 的边际收益。
- eval 输出会写入 `runs/eval/{date}.json`，该目录默认不提交。

## 样例语料策略

仓库内 `data/baguwen/` 提供少量自写样例语料，当前包括：

- 操作系统
- Python
- 大模型与 Agent
- 计算机网络
- 数据库与缓存
- 并发编程
- 系统设计
- RAG / embedding / reranker
- 机器学习基础
- Git 与 Linux 排障

这些文件仅用于个人学习和本地知识库构建，不公开再分发第三方付费或受限内容。需要扩充个人语料时，请放入：

```text
data/baguwen/private/
```

该目录已加入 `.gitignore`。

## 测试与验证

运行单元测试：

```bash
uv run pytest
```

当前验证基线：

- `uv run pytest` -> 40 passed。
- `uv run python scripts/build_index.py --data-dir data/baguwen --persist-dir .chroma/jobagent` -> 162 chunks indexed。
- `uv run python -m src.eval.retrieval_eval` -> 12 cases completed。

## 主要目录

```text
jobagent/
├── app.py                         # Streamlit UI
├── examples/                      # 示例 JD
├── scripts/build_index.py          # 构建 Chroma 索引
├── src/
│   ├── cli.py                     # CLI 入口
│   ├── agent/                     # LangGraph workflow 和节点
│   ├── rag/                       # ingest / retriever / rerank / store
│   ├── tools/                     # Tavily web search
│   ├── schemas/                   # JobInput / Evidence / PrepPlan
│   └── eval/                      # retrieval eval
├── data/baguwen/                  # 自写样例语料
├── tests/                         # 单元测试
└── plans/                         # 每日开发计划与完成记录
```

## 已知限制

- 当前 eval set 已经饱和，不能单独证明 reranker 的实际收益。
- README 记录的是本地 MVP，不包含部署、登录、多用户、数据库后台等生产能力。
- Tavily 只使用摘要搜索结果，没有抓取完整网页正文。
- Streamlit UI 仍以 demo 为主，视觉和交互还有打磨空间。
- LangSmith live smoke、截图和完整 demo 录制还在后续计划中。

## 后续计划

近期重点：

- 扫描 `n_candidates` / `top_k` / 查询前缀等检索参数。
- 补充更难的 eval case，并把评测分析写得更可信。
- 验证 LangSmith trace 和 live smoke。
- 打磨 Streamlit UI 和 README 展示材料。
- 后续视进度加入模拟面试闭环。
