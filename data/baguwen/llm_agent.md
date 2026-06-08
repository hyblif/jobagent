# 大模型与 AI Agent 系统面试八股

## RAG 系统

Q1: RAG（检索增强生成）的原理和完整流程是什么？
A: RAG 将外部知识库引入 LLM 推理，解决知识截止和幻觉问题。完整流程：(1) 离线索引——文档解析、按策略切块（固定长度/语义/Q&A），用 embedding 模型将文本转为向量，存入向量数据库（如 Chroma、Milvus）；(2) 在线检索——用户 query 经 embedding 后在向量库做 ANN 近似最近邻搜索（HNSW/IVF），召回 top-n 候选；(3) 重排——用 CrossEncoder reranker 对 (query, doc) 对打分，取高精度 top-k；(4) 生成——将召回文档拼入 prompt，LLM 基于证据生成答案并引用来源。

Q2: Embedding 向量检索的原理是什么？HNSW 算法做什么？
A: Embedding 模型（如 BGE、text-embedding-ada）将文本映射到高维向量空间，语义相近的文本向量距离近。检索时计算 query 向量与库中所有向量的相似度（余弦相似度或内积），返回最近邻。暴力搜索 O(n)，HNSW（Hierarchical Navigable Small World）是近似最近邻算法：构建多层图结构，每层连接近邻，上层稀疏下层稠密；查询从最上层出发贪心导航，复杂度近似 O(log n)，召回率与精确搜索接近（通常 >95%），是 Chroma/Milvus 的默认索引。

Q3: Reranker 的作用是什么？CrossEncoder 和 BiEncoder 的区别？
A: Reranker 对粗召回结果进行精排，提升 top-k precision。BiEncoder（双塔）：query 和 doc 独立 encode，离线预计算 doc 向量，在线只需 encode query，速度快，适合大规模召回，但 query-doc 交互弱，精度较低。CrossEncoder（交叉编码器）：将 (query, doc) 拼接后共同 encode，Attention 层充分建模两者交互，精度更高；但无法预计算，只适合对少量候选（top-20~100）打分，不能做大规模召回。RAG 典型组合：BiEncoder 召回 top-20，CrossEncoder 重排取 top-5。

## Agent 系统

Q4: Function Calling / Tool Use 的工作原理是什么？
A: LLM 在推理时可以"调用"外部工具：开发者在请求中传入工具描述（函数名、参数 schema、说明），LLM 根据用户意图决定是否调用、调用哪个工具以及传入什么参数，以结构化 JSON 格式输出工具调用请求；应用层执行实际调用，将结果返回 LLM 继续推理，直到生成最终回复。关键点：工具描述的质量直接影响调用准确性；需处理并行调用、多轮调用和调用失败的情况；JSON Schema 校验工具参数防止幻觉。

Q5: ReAct 和 Plan-and-Execute 两种 Agent 模式有什么区别？
A: ReAct（Reasoning + Acting）：交替进行"思考—行动—观察"，每步基于当前观察动态决策下一步，适合不确定性高、步骤短的任务；但长任务中容易偏离目标，token 消耗大。Plan-and-Execute：先生成完整计划（任务分解），再逐步执行；规划阶段全局视野更好，执行阶段各步骤更专注，适合长流程确定性任务；缺点是计划刚性，中途遇到意外需重新规划。LangGraph 图状态机可以灵活实现两种模式及其混合：通过条件边动态路由，通过循环实现反思和自修复。

Q6: LangGraph 状态机编排的核心优势是什么？
A: LangGraph 基于有向图（DAG + 循环）管理 Agent 工作流：(1) 显式状态管理——TypedDict State 记录所有中间结果，节点只返回增量更新，状态变更可追溯；(2) 条件边路由——`add_conditional_edges` 基于状态动态选择下一节点，自然实现分支、重试、循环；(3) 自修复循环——validate 节点检测错误后路由回 plan 节点，将错误信息注入 prompt，实现闭环自修复；(4) LangSmith 自动追踪——每个节点的输入输出、耗时、重试均可视化，调试大幅简化。相比纯函数链（LangChain Expression Language），LangGraph 更适合有状态、有循环的复杂工作流。

## LLM 基础

Q7: Attention 机制的工作原理是什么？Self-Attention 如何计算？
A: Attention 计算每个位置与其他所有位置的相关性权重，加权汇总 Value 向量。Self-Attention 步骤：(1) 输入 X 通过三个线性变换得到 Q（Query）、K（Key）、V（Value）矩阵；(2) 计算注意力分数：`Attention(Q,K,V) = softmax(QK^T / √d_k) V`，除以 √d_k 防止点积过大导致梯度消失；(3) Multi-Head Attention 并行运行多组 Attention，拼接后线性投影，捕获不同维度的语义关系。Causal Mask 遮盖未来位置，保证自回归生成。KV Cache 缓存历史 K、V，推理时复用，降低时间复杂度从 O(n²) 到 O(n)。

Q8: LLM 幻觉的成因和缓解策略有哪些？
A: 幻觉（Hallucination）指 LLM 生成看似合理但事实错误的内容。成因：训练数据中的噪声和偏见、自回归解码的概率采样、模型对知识截止日期后信息的外推。缓解策略：(1) RAG——从外部可信知识库检索并引用，减少对训练参数的依赖；(2) 强制结构化输出（JSON mode + schema 校验）——减少格式幻觉；(3) source_id 引用机制——要求 LLM 每条结论引用可验证 id，并在后处理阶段校验引用合法性；(4) 温度降低（temperature 接近 0）——减少随机采样幻觉；(5) CoT（思维链）——显式推理步骤暴露错误；(6) 自我验证循环——generate-validate-fix 闭环。

Q9: Prompt 注入攻击是什么？如何防护？
A: Prompt 注入：攻击者通过输入恶意指令覆盖或篡改系统 prompt，使 LLM 执行非预期行为（如忽略之前的指令、泄露系统 prompt、执行恶意工具调用）。直接注入：用户输入中包含"忽略之前所有指令"等内容；间接注入：通过 RAG 检索到的文档中嵌入恶意指令。防护措施：(1) 使用 OpenAI-compatible API 的 system/user 角色隔离，system prompt 不对用户可见；(2) 对用户输入做内容过滤，检测注入关键词；(3) 工具调用做参数校验和权限控制，最小权限原则；(4) 输出后处理——校验 JSON schema 和引用合法性；(5) 关键操作（写文件、执行命令）增加人工确认环节。
