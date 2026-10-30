# M4 设计 spec:单卡追平 vLLM(CUDA graph)

日期:2026-10-30
状态:已与用户对齐(四项决定逐条确认),待出实施计划
前序:M3 已收官(`m3-acceptance`,C=8 以 99.2% 带缺口收官,2.23× 预案达成)

## 0. 背景与北极星

用户北极星:**与官方 vLLM 正面比拼性能**。M3 收官数字(triton+fused 全开,vs vLLM 0.32.0 基线):

| C | vLLM | M3 实测 | vs vLLM | TPOT(vLLM / M3) |
|---|---|---|---|---|
| 1 | 57.81 | 50.39 | 87.2% | 16.92 / 19.28 ms |
| 4 | 207.39 | 163.91 | 79.0% | 17.86 / 22.59 ms |
| 8 | 354.91 | 246.37 | 69.4% | 17.84 / 24.93 ms |

追平 ⇒ TPOT 需再降 −12% / −21% / −28%。M3 归因(标注为推断,本里程碑 T1 量化):剩余缺口主因是无 CUDA graph 的逐 kernel 发射间隙 + CPU 步内逻辑;K3 kernel 本体 ~0.042ms/层 × 36 层 ≈ 1.5ms/步,占 TPOT 不到 10%。

## 1. 四项决定(用户逐条确认)

1. **M4 = 单卡追平,TP=2 挪至 M5**(与昇腾体验合并再排)。最短路径打北极星。
2. **验收线 = 逐档 ≥100% vLLM**:聚合吞吐 C=1/4/8 逐档超过官方实测。雄心目标;不达标走 §5 预案,不许自行放宽。
3. **基线口径 = 当日配对 A/B**:验收日在同机同时间窗口重跑自建 vLLM 服务(GPU 0)与引擎(GPU 1),以当日实测为基线表(替换 Task 1 归档在验收表中的地位;归档保留作历史参照)。理由:M3 观察到跨日机时漂移 ±2~3%,≥100% 判定经不起该噪声。
4. **技术主轴 = profiler 量化 → 手写 CUDA graph → 数值保持调参**。`torch.compile reduce-overhead` 否决(接管手写 triton 编译,逐位确定性/golden 门风险高,违背手搓初衷),仅作 A 失败后的兜底重议选项。

## 2. 架构

唯一新模块 `acorninfer/engine/graph_runner.py`,含 `DecodeGraphRunner`:

- **批大小桶** `{1, 2, 4, 8}`:懒捕获(首次遇到某桶:side-stream warmup → capture);B_actual pad 到所属桶。
- **静态输入缓冲**(图外原位更新,图内重放读取):`ids [Bmax,1]`、`positions [Bmax]`、`tables [Bmax,MP] int32`、`lens [Bmax] int32`、scatter 索引 `(pages, slots) [Bmax]`。
- **图内内容**:embed → 36 ×(K1 fused_add_rmsnorm → qkv GEMM → K2 rope → K3 paged attn + scatter 写页 → o_proj → K1 → MLP)→ norm → lm_head → argmax(greedy 采样在图内)。
- **每桶一张 graph + 私有内存池**:capture 冻结该桶全部中间激活分配,重放零分配。

**步协议(CPU 逻辑全部留图外)**:executor 现有流程——prepare 容量确保、tables/lens 构建、`_lens` 簿记、EOS 状态机——一行不动;仅将 `model.forward_decode` 调用点替换为「原位写静态缓冲 → `graph.replay()` → 读静态输出缓冲」。pager 页张量本就是设备上原位更新,天然图内安全。

**Padding 行**:scatter 索引指向每层预留的一页 scratch(捕获前分配,永不读);EOS 后序列继续空转、输出由 executor 屏蔽(与现状 decode 循环语义一致)。桶 {1,2,4,8} 下 C=8 满桶零浪费;非满桶批(如 5/6/7)pad 到 8,仅浪费算力不影响正确性。

**开关**:`EngineConfig.decode_graph: bool = False`(默认关 = M3 行为;与 `attention_backend`/`fused_ops` 正交);bench CLI 加 `--cuda-graph`;**eager triton 路径完整保留为 reference**——形成 SDPA → triton eager → triton graph 三级 reference 链,每级可对拍。

## 3. 数值契约

- graph 重放 = 相同算子相同形状,理论逐位。双门钉死:
  1. tiny 模型 graph vs eager(logits/采样 token)逐位;
  2. golden M4:graph 路径引擎并发输出 == 顺序参考 token 逐位(镜像 golden M3 协议:同 margin 筛选 prompt、sdpa/triton-eager/graph 三引擎对拍、fuzz 无泄漏)。
- kernel 调参纪律:profiler 数据说话;**只做数值保持类**调参(num_warps 等,调后 golden 全量复验);改归约序的(exp2、num_splits)默认不动——除非量化收益 >5% 且走 M0/M2/M3 换题协议(记录新 prompt 实测 margin)。

## 4. 任务切分(预览,细案归实施计划)

- T1 profiler 量化:torch.profiler 拆解 24.93ms(C=8)——GPU kernel 时间 vs 发射间隙 vs CPU 步内;docs 的"推断"标注升级为实测账目。
- T2 graph 基建:DecodeGraphRunner + 桶/缓冲/捕获重放生命周期 + tiny 级 graph-vs-eager 逐位单测。
- T3 executor 接线:decode_graph 开关穿透(from_config/bench)、Engine 集成、桶命中与 padding 正确性单测。
- T4 golden M4:三引擎对拍 + fuzz 无泄漏;golden M3(eager)回归复验。
- T5 kernel 调参(条件任务):仅当 T1 账目证明 kernel 时间占比值得,且按 §3 纪律。
- T6 验收:当日配对 A/B 三档 + docs + `m4-acceptance` 判定。

每任务照旧双审查(spec + 质量,Do Not Trust the Report);实现子代理同时只跑一个。

## 5. 验收门与风险预案

**中间门**:T3/T4 落地即测一次配对 A/B(预期落在 vLLM 的 95~105% 区间)——数据决定 T5 力度。
**最终门**:验收日配对 A/B,三档逐档 ≥100%。
**BLOCKED 预案**(预写,不自行放宽):若最终落在 ≥95% 且 <100%,按 M3 先例带缺口重议——条件:差距归因带 profiler 实测入档 + 已穷尽数值保持调参;由用户裁决(参考 M3:2/3 档达标 + 预案参考达成 → 带缺口收官)。

**风险清单**:
1. capture 与内存池交互:捕获在私有池,页张量/权重是捕获前常驻分配——理论上无冲突,T2 第一验证点;
2. triton kernel 图内捕获兼容性:成熟用法(vLLM 在用),K3/K1/K2 需实测;
3. 非满桶 padding 稀释 C=4~8 之间批的吞吐(仅算力浪费,正确性无虞);
4. **≥100% 是硬目标**:graph 消发射间隙,vLLM 的 kernel 也强,可能落在 95~99% → §5 预案;
5. 机时噪声:配对 A/B 消;bench 每条前 nvidia-smi 查残留(M3 纪律延续)。

## 6. 保护文件与纪律(沿用 M3)

- golden m0/m1/m2/m3、`cache/kv.py`、`engine/scheduler.py`、`engine/sequence.py`:禁改;`cache/pager.py`:本里程碑禁改(M3.5 已给足接口);`models/qwen3.py`:torch 路径与 eager triton 路径 byte-identical(graph 接线走 executor 层,不改模型);`ops/*`:仅 T5 条件性微调,按 §3 纪律。
- GPU 分工:GPU 1 开发测试(`CUDA_VISIBLE_DEVICES=1`);GPU 0 仅验收日配对 A/B 时跑 vLLM 服务。
- 实测数字才可写入 docs;容差不拍脑袋;runtime 不 import transformers。
