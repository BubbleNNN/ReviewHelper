---
title: 并行策略：DP、FSDP、TP、PP、CP、EP
summary: 每种并行切的是什么、通信发生在哪、代价是什么，以及给一个模型配并行度的思路。
tags: 并行, FSDP, Megatron, MoE
order: 40
---

## 总览

<div class="cols"><div><h4>DP / ZeRO / FSDP</h4><p>切数据（以及模型状态），每卡算完整模型。通信：梯度 all-reduce 或 reduce-scatter + 参数 all-gather。</p></div><div><h4>TP</h4><p>切每层的矩阵。通信：每层前向 2 次 all-reduce，在关键路径上。</p></div><div><h4>PP</h4><p>切层。通信：相邻 stage 点对点传激活。代价：bubble。</p></div><div><h4>CP / Ulysses</h4><p>切序列。通信：Ring 传 KV，或 all-to-all 换头。</p></div><div><h4>EP</h4><p>切 MoE 专家。通信：dispatch / combine 两次 all-to-all。</p></div></div>

总卡数 = TP × CP × PP × DP（MoE 里 EP 通常复用 DP 那一维的卡）。

## 数据并行：DDP → ZeRO → FSDP

**DDP**：每卡一份完整模型，各吃不同的数据，反向时 all-reduce 梯度（bucket 化、与反向重叠，见“集合通信”）。简单高效，但每卡都要放下 $16\Psi$ 字节的模型状态。

**ZeRO** 把模型状态切开，显存公式见“显存账”一篇。**FSDP** 是 PyTorch 里的 ZeRO-3 实现：

<div class="flow"><span class="f-infra">all-gather 第 i 层参数</span><i>→</i><span>前向计算</span><i>→</i><span>释放完整参数</span><i>…</i><span class="f-infra">反向再 all-gather</span><i>→</i><span>算梯度</span><i>→</i><span class="f-rl">reduce-scatter 梯度</span><i>→</i><span>释放</span></div>

- **prefetch**：算第 $i$ 层时提前发起第 $i+1$ 层的 all-gather（`forward_prefetch`、`backward_prefetch`），把通信藏在计算后面。
- **分片策略**：`FULL_SHARD`（ZeRO-3）、`SHARD_GRAD_OP`（前向后不释放参数，近似 ZeRO-2）、`NO_SHARD`（等价 DDP）、`HYBRID_SHARD`（节点内 FULL_SHARD、节点间复制，即 HSDP，跨节点只做梯度 all-reduce）。
- **FSDP1 vs FSDP2**：FSDP1 把一个 wrap 单元里的所有参数拼成一个一维 `FlatParameter` 再切；FSDP2（`fully_shard`）对每个参数用 DTensor 沿第 0 维切。FSDP2 去掉了 FlatParameter，显存行为更可预测，更容易和 TP 等其它 DTensor 并行组合，也方便对单个参数做冻结、换精度。verl 的 `strategy=fsdp2` 就是它。

> [!Q]
> **FSDP 的 wrap 粒度怎么选？**
> 一般以 Transformer block 为单位 wrap。粒度太粗（整个模型一个单元），all-gather 出来的完整参数太大，省不了显存；粒度太细，通信次数多、每次消息小，延迟占主导、难以和计算重叠。

## 张量并行（Megatron TP）

以 MLP $Z = \text{GeLU}(XA)\,B$ 为例，TP 度为 $t$：

1. $A$ **按列切**：$A = [A_1, \dots, A_t]$，每卡算 $Y_i = \text{GeLU}(XA_i)$。GeLU 是逐元素的，不需要通信。
2. $B$ **按行切**：$B = [B_1; \dots; B_t]$，每卡算 $Y_i B_i$，结果是部分和，需要一次 **all-reduce** 得到 $Z = \sum_i Y_i B_i$。

Attention 同理：Q/K/V 投影按头列切（每卡负责 $n_{\text{heads}}/t$ 个头），输出投影按行切，最后一次 all-reduce。所以**每个 Transformer 层前向 2 次 all-reduce，反向 2 次**，每次通信量是 $b \cdot s \cdot h$ 个元素。

其它细节：

- Megatron 用一对共轭算子 $f$/$g$ 描述：$f$ 前向恒等、反向 all-reduce；$g$ 前向 all-reduce、反向恒等。
- Embedding 和输出层按**词表**切（vocab parallel），cross-entropy 也做成并行版本，不用把完整 logits 收集到一张卡。
- 头数、FFN 维度、词表大小都要能被 $t$ 整除。

> [!KEY]
> TP 的通信频繁、在关键路径上、每次量不小，必须走 NVLink。所以 TP 通常 ≤ 8，限制在一个节点内。切太细还会让每卡的 GEMM 变小、Tensor Core 吃不饱。

### Sequence Parallel（Megatron SP）

TP 只切了 GEMM，LayerNorm 和 Dropout 的输入在 $t$ 张卡上是**重复**的完整激活。SP 把这些区域沿序列维切开，并把原来的 all-reduce 拆成 reduce-scatter（进入 SP 区域）+ all-gather（进入 TP 区域）。通信量不变，但这部分激活显存降为 $1/t$。

## 流水线并行（PP）

把层切成 $p$ 个 stage，每个 stage 放在不同的卡上，一个 batch 拆成 $m$ 个 micro-batch 依次流过。

- **GPipe**：所有 micro-batch 先做完前向，再统一做反向。设每个 micro-batch 在一个 stage 上的前后向时间为 $t_f + t_b$，空闲（bubble）时间为 $(p-1)(t_f+t_b)$，占总时间的比例为

$$\text{bubble} = \frac{p-1}{m+p-1}$$

（Megatron 论文写成 bubble 时间与理想计算时间之比 $(p-1)/m$，两种写法都对，说清楚分母即可。）GPipe 要同时存 $m$ 个 micro-batch 的激活。

- **1F1B**：warmup 阶段做几个前向后，每个 stage 交替做一次前向、一次反向，及时释放激活。bubble 和 GPipe 一样，但同时在飞的 micro-batch 最多 $p$ 个，**激活显存与 $m$ 无关**。
- **Interleaved 1F1B**（virtual pipeline）：每张卡持有 $v$ 段不相邻的层，bubble 降为约 $(p-1)/(v\,m)$，代价是点对点通信次数变为 $v$ 倍。
- **Zero Bubble / DualPipe**：把反向拆成“对输入求梯度（B）”和“对权重求梯度（W）”，W 可以推迟去填 bubble；DeepSeek-V3 的 DualPipe 从两端同时注入 micro-batch，并让计算和 all-to-all 通信重叠。

> [!Q]
> **PP=8，想把 bubble 压到 10% 以下，micro-batch 数至少多少？**
> 用 $(p-1)/(m+p-1) < 0.1$，得 $m > 9(p-1) = 63$，即 $m \ge 64$。如果用比值 $(p-1)/m < 0.1$，则 $m > 70$。经验法则是 $m \ge 4p$。

动手：`exercises/system/pipeline_bubble` 模拟 GPipe / 1F1B 调度并数 bubble。

## 长序列：Context Parallel 与 Ulysses

序列很长时，单卡放不下一个样本的激活，就要沿序列维切。

**Ring Attention / Context Parallel（CP）**：每卡持有一段 Q、K、V。K、V 块沿环依次传给下一张卡，每卡用自己的 Q 和收到的 KV 块算部分 attention，用 online softmax 的方式合并（和 FlashAttention 同一个技巧）。传输与计算可以重叠。causal mask 下前面的卡活少、后面的卡活多，常用 zigzag 切法（第 $i$ 张卡拿第 $i$ 块和第 $2\,cp-1-i$ 块）来均衡。

**DeepSpeed-Ulysses**：输入按序列切 `[s/p, h]`。进入 attention 前做一次 **all-to-all**，变成按头切 `[s, h/p]`，每卡拿完整序列、部分头，直接用任意 attention kernel 算；算完再 all-to-all 换回来。

| | Ring / CP | Ulysses |
|---|---|---|
| 通信 | 点对点传 KV，可与计算重叠 | 每层 attention 前后各一次 all-to-all |
| 并行度上限 | 不受头数限制 | 不能超过头数（GQA 下受 KV 头数约束，或需复制 KV） |
| attention kernel | 需要支持分块合并 | 任意现成 kernel |

verl 的 FSDP 后端用的是 Ulysses：`ulysses_sequence_parallel_size`。

## 专家并行（EP）

MoE 层把不同专家放到不同卡上：

<div class="flow"><span>router 选 top-k 专家</span><i>→</i><span class="f-infra">all-to-all dispatch<small>token 发到专家所在卡</small></span><i>→</i><span>专家 FFN</span><i>→</i><span class="f-infra">all-to-all combine<small>结果发回原卡</small></span><i>→</i><span>加权求和</span></div>

- attention 等非专家部分照常做 DP/TP，专家部分做 EP。
- **负载均衡**：热门专家所在的卡会成为瓶颈。常见手段：auxiliary load-balancing loss（Switch / GShard）、capacity factor 加 token drop、DeepSeek-V3 的 aux-loss-free 方法（给每个专家的路由分数加一个可调 bias）。
- 动手：`exercises/model/moe_router`（top-k 路由与负载均衡损失）。

## 怎么给一个模型配并行度

1. **先看能不能只用 FSDP/ZeRO-3 + DP**：模型状态 $16\Psi/N$ 加激活放得下，就用它，最简单、MFU 也不差。几十 B 以内的 dense 模型、卡数足够时常见。
2. **放不下或者扩展性不够时加 TP**：节点内 TP=2/4/8，降低每卡参数和激活。
3. **还放不下加 PP**：跨节点切层，保证 $m \ge 4p$。
4. **序列太长加 CP / Ulysses**。
5. **MoE 加 EP**。
6. **剩下的卡给 DP**，并核对 global batch = micro_bs × $m$ × DP 是否符合训练要求。

放置顺序从内到外：TP（最密集，放节点内）→ CP/EP → PP → DP（最稀疏，放最外层）。

参考：Llama 3 405B 预训练在 16K 张 H100 上用 TP=8、PP=16、DP=128，长上下文阶段再加 CP=16，报告的 BF16 MFU 在 38%–43%（Llama 3 论文；具体配置待核实）。

> [!Q]
> **RL 训练时，训练和推理用的并行方式一样吗？**
> 一般不一样。训练侧追求显存和吞吐，常用 FSDP（+ Ulysses）或 Megatron 的 TP/PP/EP；rollout 侧追求低延迟、高吞吐，常用小 TP（甚至 TP=1）多副本。所以每次更新后要把权重从训练布局**重排（reshard）**成推理布局，这正是 verl 3D-HybridEngine / checkpoint engine 解决的问题。

> [!WARN]
> 不要把 Megatron 的 Sequence Parallel 和 Ulysses / CP 混为一谈：前者是 TP 的配套，只切 LayerNorm/Dropout 区域的激活，attention 本身仍然看完整序列；后两者才是真正把 attention 的序列维切开。

动手：`exercises/system/tp_linear` 验证 column/row parallel 的矩阵乘和完整矩阵乘结果一致。
