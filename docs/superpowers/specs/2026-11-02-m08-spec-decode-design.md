# M8 设计:投机解码(n-gram 草稿 + 贪心验证,eager v1)

> 状态:设计已获批(D1–D7 全按推荐定案,2026-11-02);本文件为定案文本。
> 前置:M7 终账(`docs/benchmarks.md` M7 终账节,HEAD `dd90d72`,tag `m7-acceptance`)。

## 0. 背景与动机

"墙物理"三部曲收官:M6 证明 decode 被权重流钉死(~950GB/s 天花板),M7 把字节压到 7.56GB/步(W8A16,TPOT 11.5ms),**M8 摊薄读次**——一次前向验证 K+1 个位置,让每份权重流产出多个 token。与 M7 复利:若 E[接受+1] ≈ 2,有效 TPOT 冲 6–9ms。

选型 = PLD 式 n-gram 草稿(免 draft 模型、零额外显存、纯 CPU 查表)+ 贪心验证。**贪心投机在数学上无损**:每个被接受的 draft token 恰是目标模型该位置的 argmax,而非投机贪心链是确定性的——spec 输出与非投机贪心逐位一致。这一可证明性质成为本里程碑最硬的正确性锚(全场唯一拿"输出不变"当验收的性能特性)。

诚实前提:投机对 workload 敏感。聚合/续写型(输出复制输入)接受率高;聊天型(每 token 都是新信息)接受率趋近 0,此时 eager verify 税使其小幅**输给** W8 graph 基线。本里程碑把两种面向都如实计量,不粉饰。

## 1. 验收锚

| 锚 | 门 | 说明 |
|---|---|---|
| **A1 无损性(硬门)** | spec 贪心 ≡ 非投机贪心**逐位** | tiny 模型全量对拍 + 8B 抽样请求(多 workload);任何一位不同 = BLOCKED |
| **A2 加速(硬门)** | wiki 续写 workload,C=1,tok_s ×≥1.4 vs **同模式 eager W8 基线** | 同模式对齐(均为 eager);另附 vs 生产 graph W8 的比值(如实,预估 ×1.2–1.6) |
| **A3 workload 敏感性(如实)** | chat 型 workload 不设门 | 允许 <1×,归因 = 接受率;两面向矩阵进 docs |
| **A4 缺省不变(硬门)** | `--spec` 缺省关,非投机路径逐字节 | 全套件零换金 |

接受率定义:`E[接受+1] / verify 步`(逐 workload 报告,含 tokens/step 分布)。A2 的 ×1.4 对应 E[接受+1] ≥ ~1.45(wiki 型 PLD 文献带 1.8–2.8,留有余量);若 1.4 未达:如实裁,**参数扫(N∈{1,2,3} × K∈{2,4,6})作为施工期数据先行,锚在选定配置上判**——选定配置 = 扫描后 docs 记录的缺省,不允许仪式当天换锚。

## 2. 架构

### 2.1 草稿器 `acorninfer/spec/ngram.py`

- 上下文索引:dict[tuple[int,...], list[int]]——键 = 上下文中出现的 N-gram,值 = 其后继 token(按出现序);admit 时从 prompt 一次建索引,每接受 token 后增量更新(键值对 append,O(1))。
- `propose(last N tokens) -> [K candidates]`:查表取后继列表;不足 K 用固定 filler 补齐(如 token id 0,确定性;填充位置照常参与前向,但其 logits 被接受逻辑忽略,无正确性风险)。**确定性**(同上下文同输出)。
- 纯 CPU python;上下文 ≤512 token,索引内存有界(≤512×K 条目量级);draft 耗时目标 <100µs/步(实测入账)。

### 2.2 verify 前向(m = K+1,全批统一)

- `models/qwen3.py::forward_verify`(新变体):embed m 个 token → 每层 attention 以 m 个 query 位读分页历史 + **窗口内因果掩码**(位置 i 可见历史 + 0..i−1)→ 投影分派复用现有 linear/quant 分派 → lm_head 输出 m 位置 logits(`.float()`)。
- verify attention kernel:`ops/paged_verify.py`(**新文件**,从 `ops/paged_attention.py` 的结构派生——**只读不改**,保护原件);语义 = 单新位 kernel 的 m 位推广 + 窗口内因果;对拍参考 = 同输入的稠密参考实现(TOL 新标定)。
- 接受逻辑(引擎级,`engine.py`):`argmax(logits[i]) == draft[i]` 逐位比,取最长前缀 a;全中则 bonus = `argmax(logits[K])`;输出 = 接受 token 序列(+bonus);`max_tokens`/eos 逐 token 检查,越界即截断(现有 Sequence 单 token 方法组合调用,sequence.py 不动)。
- **batching:全批统一 m**(D2 填充方案);`scheduler.py` 不动——草稿状态(每 seq 索引器)由引擎侧 `dict[seq_id]` 管理。
- 验证步的每 seq 输出 = 多 token:`TokenOutput` 列表按 seq 内顺序发射;server SSE 逐 token 流式(顺序契约测试钉住)。

### 2.3 KV 游标复用(D3,pager 最小解禁)

- verify 一次写 m 个位置的 KV 于游标 `[len, len+m)`,接受 a 个 → `lens = len+a+1`;**下一步写游标从 len' 起复用死槽**(重写覆盖,页表项已在)——零泄漏、零分配器回滚、free-list 语义不变。
- 现有 triton attention 按 lens 读页 → 死槽天然不可见(正确性由构造保证)。
- pager.py 解禁条款(唯一):`ensure_append_capacity` 支持增量 n>1(缺省 1 语义逐字节不变)+ 多槽写游标;`cache/kv.py` 先读后定(纯 pager 层足够则不动);跨 block 边界的游标续接是必测边(页分配照常走 allocator)。

### 2.4 v1 执行模式(D1)

- **eager verify**(不走 graph):`--spec` 生效时该引擎实例 spec 模式全 eager;`spec × --cuda-graph` 显式互斥报错(v1);graph verify = M9 候选。
- 互斥面:`spec × tp 2`、`spec × 非贪心采样`(temperature>0 或 top_p<1)——构造期 ValueError(spec §D4:无模型草稿分布,采样域数学不成立)。
- 缺省 `--spec-ngram 0`(关)= 现行为逐字节;开启 = `--spec-ngram 2 --spec-num 4`(m=5)。

## 3. 保护面

禁改照旧:golden m0–m7、`cache/kv.py`(视 §2.3,先读)、`ops/paged_attention.py`(**只读派生**)、`engine/scheduler.py`、`engine/sequence.py`、`engine/graph_runner.py`(eager v1 无需触碰)、`tests/triton_utils.py`。

本程解锁/新建:
- `models/qwen3.py`:新增 `forward_verify` 变体 + 调用点(投影/quant 分派复用,不动既有 4 前向);
- `cache/pager.py`:§2.3 最小解禁(唯一保护文件解禁项);
- `engine/engine.py` + `executor.py` + `server.py` + `acorninfer/config.py`:spec 步路径与配置面;
- 新建:`acorninfer/spec/`、`ops/paged_verify.py`、`scripts/`(语料切分)、`tests/test_spec_*.py`。

## 4. 风险表

| # | 风险 | 对策 |
|---|---|---|
| R1 | wiki 锚 E[接受+1] < 1.45 | 施工期参数扫(N×K)先行、选定配置锚判(docs 记录);仍不达 = 如实 NO-GO(特性保留、锚挂账),workload 矩阵照出 |
| R2 | verify kernel 数值错 | 稠密参考 TOL 新标定 + A1 无损性双保险(任何数值偏差都会打破逐位一致) |
| R3 | KV 游标跨 block 边界/死槽复用 bug | 页表断言 + KV 对拍 + A1(泄漏/错读必然破坏输出一致) |
| R4 | chat 型负收益 | A3 如实呈现;缺省关;docs 给 workload 选择指引 |
| R5 | 草稿器开销/索引内存 | <100µs/步 + 有界性实测入账 |
| R6 | 多 token SSE 顺序 | 顺序契约测试(server 层) |
| R7 | vLLM 对照腿配置(ngram spec) | 仪式可选腿:vLLM `--speculative-config` ngram(0.32 支持性先验证,不行则如实缺席) |

## 5. 非 goal

draft-model 系(Eagle/Medusa)、tree attention、自适应 k(M9 候选:低接受率自动降 m)、graph verify(M9)、采样域投机、TP×spec、W4/连续到达(台账另列)。

## 6. 仪式与收口

- 语料:wiki 续写 prompt = `data/wikitext-2-raw/wiki.test.raw` 切 128-token 块(M7 缓存复用),out=32;chat 型 = 现行默认 bench 语料。语料切分脚本入 `scripts/`(确定性 seed,可复现)。
- 仪式腿(GPU 0,M5/M6 配方):vLLM bf16 / AcornInfer W8 graph(生产参照)/ W8 eager(同模式基线)/ **W8 spec eager(wiki + chat 双 workload)**;可选 vLLM ngram spec 腿(R7)。
- 全量回归:fast(基线 217 + 新增)/ slow 17 / tp 13 零换金;A1/A4 硬门证据入档。
- 判定行(如实):A1–A4 逐条;接受率矩阵;参数扫记录;段 B 式诚实条款(锚不达照写)。
- tag/push 留用户。
