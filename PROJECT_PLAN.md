# jobagent MVP 项目计划

## 1. 项目定位

jobagent 是一个面向求职准备场景的 AI Agent 应用：输入目标公司、岗位和 JD 后，系统自动做联网岗位调研，结合本地八股/面经知识库检索，生成一份带来源引用的结构化面试备战计划。

本项目的第一目标是成为可演示、可写进简历、可在面试中讲清楚工程链路的作品。MVP 以本地演示为主，不追求一开始做成完整 SaaS 产品。

关键时间线：

- 2026-06-08 到 2026-06-14：跑通 MVP 主链路。
- 2026-06-15 到 2026-06-21：完善 UI、评测、鲁棒性和 README。
- 2026-06-22 到 2026-06-28：视进度增加模拟面试闭环、录制 demo、打磨简历材料。
- 2026-06-29 到 2026-06-30：简历定稿并投递。

## 2. 已确定的 MVP 边界

本期要做：

- 输入：公司、岗位、JD 文本。
- 联网调研：使用 Tavily API 搜索公司和岗位相关的面试流程、面经、岗位信息。
- RAG：导入公开八股/面经 markdown 语料，本地向量库召回候选片段后，用 reranker 重排取 top-k。
- 计划生成：生成结构化备战计划，包括面试流程概览、高频考点、八股清单、答案要点、准备建议和引用来源。
- 展示入口：同时提供 CLI 和 Streamlit UI，二者复用同一套核心 service/workflow。
- 本地结果保存：每次生成的计划保存为 JSON 和 Markdown，方便复现 demo。
- 可观测：接入 LangSmith，全链路 trace 每个节点的输入输出与重试（仅 `LANGSMITH_TRACING=true` 时启用）。

本期不做：

- 不做登录、多用户、权限、数据库后台。
- 不做简历解析和账号投递自动化。
- 不抓取付费面经原文，不在仓库公开保存爬取正文。
- 不做模型微调。
- 不把云部署作为 MVP 验收项。
- 模拟面试、闪卡、PPT、完整 NotebookLM 式知识管理放到后续增量。

## 3. 技术选型定稿

| 模块 | 选型 | 决策 |
| --- | --- | --- |
| 语言 | Python 3.11 | 和目标开发速度、生态、个人主力匹配。 |
| 依赖管理 | uv + pyproject.toml + uv.lock | 主路径使用 uv 管理依赖并提交锁文件；`requirements.txt` 由 `uv export` 生成，作为无 uv 环境的 fallback。 |
| UI | Streamlit | 本地演示最快，避免 React 前端消耗时间。 |
| Agent 编排 | LangGraph | intake/research/retrieve/rerank/plan/validate 有状态 workflow；validate 失败回 plan 自修复，构成真实 cycle，而非线性套壳。 |
| 检索层 | chromadb 直连 | 不引入 LlamaIndex，LangGraph node 直接调 Chroma，职责最薄、无双生态适配层；自写 ingest/retrieve 也更能体现 RAG 工程深度。 |
| 向量库 | Chroma | 本地持久化，零运维，适合 MVP demo。 |
| Embedding | BAAI/bge-small-zh-v1.5 | 本地轻量中文 embedding；支持环境变量切换到本地模型路径；升级路径预留 bge-m3。 |
| Reranker | BAAI/bge-reranker-v2-m3 | 通过 sentence-transformers `CrossEncoder` 实现，对召回候选重排，提升 precision；eval 展示加/不加 reranker 的 top-k 对比。 |
| LLM | DeepSeek OpenAI-compatible API | 默认使用 DeepSeek 服务，模型名通过环境变量配置；强制 JSON 输出。 |
| 搜索 | Tavily advanced 摘要模式 | MVP 只接 Tavily 单源，不额外抓完整网页正文。 |
| 输出格式 | JSON -> Markdown | LLM 先生成结构化 JSON，代码校验后渲染为 Markdown/UI。 |
| 可观测 | LangSmith | LangGraph 自动 instrument，近零代码看每个 node 输入输出与重试；仅 `LANGSMITH_TRACING=true` 时启用，不影响离线/CI。 |

## 4. 核心架构

```text
用户输入
  |
  v
Streamlit UI / CLI
  |
  v
LangGraph workflow
  |
  +-- intake: 解析公司、岗位、JD，抽取检索关键词
  |
  +-- research: Tavily 搜索岗位/公司面试相关信息
  |
  +-- retrieve: chromadb 从本地知识库召回八股/面经候选片段
  |
  +-- rerank: bge-reranker-v2-m3 对候选重排，取 top-k 作为 RAG evidence
  |
  +-- plan: DeepSeek 汇总 web evidence + RAG evidence，生成结构化 JSON
  |
  +-- validate: 校验 schema + source_id 存在性
  |     |
  |     +-- 失败且有重试额度 --> 回 plan（错误回灌，自修复 cycle）
  |     |
  |     +-- 通过 --> render/save: 渲染 Markdown、保存运行结果
```

职责边界：

- LangGraph 负责编排状态流转、节点调用，以及 validate->plan 的自修复重试环（这是它相对纯函数链的存在理由）。
- chromadb 只负责本地向量持久化与召回；检索/重排逻辑在 `src/rag/` 自有薄层，不引入 LlamaIndex。
- Tavily evidence 和 RAG evidence 使用统一证据结构，供 plan 节点引用。
- Streamlit 和 CLI 不直接实现业务逻辑，只调用统一 workflow/service。

## 5. 数据与引用策略

语料策略：

- 仓库只提交少量自写样例语料和来源说明。
- 大量公开八股/面经 markdown 放在 `data/baguwen/`，本地使用，加入 `.gitignore`。
- README 注明：语料仅用于个人学习和本地知识库构建，不公开再分发第三方原文。

本地落盘：

- Chroma 索引目录：`.chroma/jobagent/`。
- 运行结果目录：`runs/{timestamp}/`。
- 每次生成保存：
  - `plan.json`
  - `plan.md`
  - 可选 `evidence.json`

引用规则：

- 每条 web evidence 和 RAG evidence 都必须有稳定 `id`。
- 计划 JSON 中的关键结论必须绑定 `source_ids`。
- 渲染 Markdown 时把 `source_ids` 转成引用列表。
- 如果 LLM 生成不存在的 `source_id`，校验层应报错并触发重试或提示失败。

## 6. 主要接口约定

环境变量：

```bash
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash

TAVILY_API_KEY=

EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
EMBEDDING_MODEL_PATH=

RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANKER_MODEL_PATH=

CHROMA_PERSIST_DIR=.chroma/jobagent

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=jobagent
```

CLI：

```bash
python scripts/build_index.py --data-dir data/baguwen --persist-dir .chroma/jobagent
python -m src.cli plan --company "阿里巴巴" --role "Agent 工程师" --jd-file examples/jd_agent_engineer.txt --out runs/demo
```

Streamlit：

```bash
streamlit run app.py
```

核心数据类型：

```python
class JobInput:
    company: str
    role: str
    jd_text: str

class Evidence:
    id: str
    source_type: str  # "web" | "kb"
    title: str
    url_or_path: str
    excerpt: str
    score: float | None

class PrepPlan:
    interview_overview: list
    focus_areas: list
    high_frequency_topics: list
    q_and_a: list
    action_plan: list
    citations: list
```

## 7. 建议仓库结构

```text
jobagent/
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py
├── examples/
│   └── jd_agent_engineer.txt
├── scripts/
│   └── build_index.py
├── src/
│   ├── cli.py
│   ├── agent/
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── state.py
│   ├── rag/
│   │   ├── ingest.py        # markdown 按 heading/Q&A 切块
│   │   ├── retriever.py     # 召回 + rerank 组合
│   │   ├── rerank.py        # bge-reranker CrossEncoder
│   │   └── store.py         # chromadb 持久化客户端
│   ├── tools/
│   │   └── web_search.py
│   ├── prompts/
│   │   └── plan_prompt.md
│   ├── schemas/
│   │   └── plan.py
│   └── eval/
│       └── retrieval_eval.py
├── data/
│   └── baguwen/
└── tests/
```

## 8. 测试与验收标准

测试计划：

- 单元测试：
  - JD intake / 关键词提取。
  - Tavily response 解析。
  - Chroma tiny corpus 召回。
  - reranker 重排顺序。
  - plan JSON schema 校验。
  - `source_id` 存在性校验 + 自修复重试（mock 非法 JSON -> 修复）。
  - Markdown 渲染。
- 集成测试：
  - 使用仓库内样例语料构建临时索引。
  - mock Tavily 和 LLM，跑完整 workflow（覆盖 validate->plan 重试路径）。
  - 验证输出包含计划结构和可解析引用。
- 可选 live smoke：
  - 当存在 `DEEPSEEK_API_KEY`、`TAVILY_API_KEY` 且 `RUN_LIVE_TESTS=1` 时，跑一次真实端到端调用。
- Retrieval / Rerank eval：
  - 准备 10 个固定面试问题。
  - 统计**加/不加 reranker** 的 top-3 / top-5 命中率。
  - README 展示对比结果和改进方向。

验收标准：

- 本地输入一个真实 JD 后，3 分钟内生成完整备战计划。
- 输出包含面试流程、高频考点、八股问答、行动计划。
- 关键结论均带可追溯引用。
- CLI 和 Streamlit 都能复用同一 workflow 跑通。
- README 能让面试官快速理解：问题、架构、技术选型、演示方式、评测指标和 Future Work。

## 9. 执行顺序

Day 1：

- 初始化项目结构、虚拟环境、`.env.example`（含 LangSmith 变量）、`requirements.txt`。
- 跑通 DeepSeek hello world，确认 LangSmith trace 能记录。
- 准备少量样例 JD 和自写八股 markdown。

Day 2：

- 实现 `rag/ingest.py`（按 heading/Q&A 切块）和 `scripts/build_index.py`。
- 跑通 bge-small-zh-v1.5 embedding + chromadb 持久化。

Day 3：

- 实现 `rag/retriever.py` 和 `rag/rerank.py`。
- 输入一个面试问题，能返回召回 + 重排后的八股片段和引用 metadata。

Day 4：

- 实现 `tools/web_search.py`。
- 跑通 Tavily advanced 搜索和 evidence 归一化。

Day 5：

- 实现 LangGraph `state.py`、`nodes.py`、`graph.py`。
- 串起 intake、research、retrieve、rerank 节点。

Day 6：

- 实现 plan 节点、JSON schema 校验、`source_id` 存在性校验，以及 validate->plan 自修复重试环。
- 实现 Markdown 渲染，CLI 端到端生成一份 `plan.json` 和 `plan.md`。

Day 7：

- 实现 Streamlit 最小界面。
- 补 README 初版、架构图、LangSmith trace 截图、retrieval/rerank 指标、运行说明和已知限制。

## 10. 后续增量

优先级 A：

- 模拟面试评分闭环：出题、作答、基于知识库评分、反馈弱点、记录下一题方向（复用现有 LangGraph 图，新增循环节点）。
- 扩充 eval：更大题库、加入答案质量/引用正确率等指标。
- 更强的失败降级：多轮 JSON 修复、检索为空时的兜底策略。

优先级 B：

- 闪卡生成。
- PPT/复习提纲导出。
- 更完整的本地知识库管理。
- Streamlit Cloud / HuggingFace Spaces 部署适配。

暂不进入 MVP：

- 多用户账号体系。
- 长期数据库持久化。
- 付费面经爬取。
- 简历自动投递。
- 模型训练或微调。
