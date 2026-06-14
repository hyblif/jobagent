# LLM RAG 工程面试八股

## RAG 流程

Q1: 一个完整 RAG 系统包含哪些模块？
A: 完整 RAG 通常包含文档采集、清洗解析、切块、embedding、向量索引、在线检索、重排、prompt 组装、LLM 生成、引用校验和评估监控。离线链路决定知识质量，在线链路决定召回和答案质量。面试时要能把“索引构建”和“用户查询”两条链路分开讲。

Q2: RAG 相比纯 LLM 直接回答有什么优势？
A: RAG 能引入外部新知识、降低幻觉、让答案可追溯、减少模型微调成本，并能按业务知识库快速更新。缺点是系统复杂度上升，检索质量会成为上限，召回错误会把无关信息带入 prompt。它适合知识变化快、需要引用来源的场景。

Q3: RAG 常见失败模式有哪些？
A: 常见失败包括文档解析丢信息、切块过大或过小、embedding 不适配领域、召回不到正确块、reranker 排错、prompt 塞入噪声、LLM 忽略证据或编造引用。排查要按链路分段看：先看 gold chunk 是否在库里，再看召回，再看重排，再看生成。

Q4: 什么是 query rewrite？
A: query rewrite 是把用户原始问题改写成更适合检索的查询，例如补全实体、拆分多跳问题、生成同义问法、提取技术关键词。它能提升召回率，但如果改写偏离原意会引入噪声。面试里可结合岗位 JD，把角色和技术栈加入本地知识库查询。

Q5: 什么是 hybrid search？
A: hybrid search 通常结合向量语义检索和关键词检索 BM25。向量检索擅长语义相似，BM25 擅长精确术语、代码名和专有名词。二者结果可用加权分数、RRF 或 reranker 融合。技术面试中可说明它能缓解 embedding 对稀有关键词不敏感的问题。

## 切块与索引

Q6: 文档切块为什么重要？
A: 切块决定检索单元的语义完整性和上下文噪声。块太小可能缺上下文，块太大容易超过 prompt 预算并稀释相关性。常见策略有固定长度、按标题层级、按段落、按 Q&A、语义切块和带 overlap 的滑窗。面试回答要结合数据形态选择策略。

Q7: Q&A 型语料应该如何切块？
A: Q&A 语料适合一题一块，问题和答案放在同一个 chunk，metadata 记录来源、标题和小节。这样用户问相似问题时能直接召回完整答案，避免问题和答案被切到两个块。若答案过长，可按小标题二次切分，但要保留问题前缀。

Q8: overlap 的作用是什么？
A: overlap 在相邻块之间保留一段重复文本，减少边界切割导致的信息丢失。它适合连续长文，但对 Q&A 语料可能不需要或应很小，因为重复问题会增加近重复块。overlap 越大召回更稳但索引膨胀、噪声也增加。

Q9: metadata 在 RAG 中有什么用？
A: metadata 用于过滤、展示引用、调试和评估。例如 source、title、heading、更新时间、权限、业务线。在线检索可以按 metadata filter 限定范围，生成时可展示来源。没有 metadata 的 RAG 很难做可靠引用和问题定位。

Q10: 如何处理文档更新？
A: 可用内容哈希判断文档是否变化，对变化文档删除旧 chunk 并重新写入新 chunk；chunk id 应稳定或可追溯；索引构建要可重复运行。大规模场景需要增量队列、版本号、软删除和回滚。面试中要强调避免重复 chunk 和脏索引。

## Embedding 与 Reranker

Q11: embedding 模型如何选择？
A: 选择要看语言、领域、维度、速度、部署成本和评测结果。中文场景可考虑 BGE 系列，多语言可考虑 bge-m3 或商业 embedding。不能只看模型榜单，要用自己的查询集评估 top-k 命中率、延迟和资源消耗。

Q12: 向量相似度常见指标有哪些？
A: 常见指标有余弦相似度、内积和欧氏距离。若向量已归一化，余弦和内积排序等价。选指标要和 embedding 模型训练目标一致，也要和向量库索引配置一致，否则分数含义会偏。Chroma、Milvus、FAISS 都支持不同距离配置。

Q13: 为什么需要 reranker？
A: BiEncoder 向量召回速度快但 query-doc 交互弱，容易把语义大致相关但不能回答问题的块排前面。CrossEncoder reranker 对少量候选做深度交互打分，能显著提升 top-k precision。典型做法是召回 top-20 到 top-100，再重排取 top-3 到 top-8。

Q14: reranker 的代价是什么？
A: reranker 无法预计算文档表示，在线要对每个 query-doc pair 前向计算，延迟和成本随候选数线性增长。候选太多会慢，候选太少会漏召回。工程上要调 n_candidates、top_k、批处理大小和模型大小，必要时缓存高频查询结果。

Q15: 如何评估 reranker 是否值得？
A: 准备固定查询集和标注相关文档，比较加/不加 reranker 的 top-1、top-3、top-5 命中率和 MRR，同时记录延迟。若相关性提升小但延迟大，可能需要优化查询、语料或 embedding，而不是盲目上 reranker。

## 生成与评估

Q16: RAG prompt 应包含哪些内容？
A: prompt 应包含任务说明、输出格式、用户问题、检索证据、引用规则和拒答/不确定策略。证据要有稳定 id，要求模型只引用已提供 id。结构化输出场景还要给 JSON schema 或字段说明，方便后处理校验。

Q17: 如何防止 LLM 编造引用？
A: 给每条 evidence 稳定 id，在 prompt 中明确 source_ids 只能来自证据列表；生成后用代码校验所有 source_id 是否存在；不存在则重试或报错；渲染时从 evidence map 回填引用详情。不能只靠 prompt 约束，必须有后处理校验。

Q18: RAG 评估指标有哪些？
A: 检索层看 recall@k、precision@k、MRR、nDCG；生成层看答案正确性、引用支持率、忠实度、完整性和格式通过率；系统层看端到端延迟、失败率和成本。MVP 可以先做小规模人工标注的 top-k 命中率，再逐步扩展。

Q19: 如何排查“答案很泛泛”的问题？
A: 先看是否有高质量 evidence；如果 evidence 为空或不相关，优化 query、语料、embedding、reranker；如果 evidence 正确但答案泛泛，优化 prompt、输出 schema、few-shot 和引用要求；如果模型忽略证据，降低温度并增加校验。分层排查比直接换模型有效。

Q20: 面试中如何讲自己的 RAG 项目亮点？
A: 可以从工程链路讲：LangGraph 编排 intake/research/retrieve/rerank/generate/validate/render；本地 Chroma + BGE embedding；CrossEncoder reranker 提升 precision；schema 校验和 source_id 检查降低幻觉；warnings 让 Tavily、KB、LLM 失败可见；eval 用 top-k 命中率证明优化有效。
