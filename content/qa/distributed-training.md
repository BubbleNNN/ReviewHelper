---
title: 分布式训练
summary: 并行策略、通信、显存与 MFU 估算的高频问题。
tags: Infra, 并行, 显存
order: 30
---

## DDP 和 `nn.DataParallel` 有什么区别？为什么都用 DDP？

- `DataParallel` 是单进程多线程：每步把模型复制到各卡、把输入 scatter 出去、把输出 gather 回 GPU0 算 loss，GPU0 负载和显存明显偏高，还受 Python GIL 限制。
- DDP 是每卡一个进程，各自前向反向，只在反向时 all-reduce 梯度，并且按 bucket 和反向计算重叠。
- DDP 支持多机，扩展性好；`DataParallel` 只能单机，基本已被弃用。

## Ring all-reduce 每张卡要发送多少数据？为什么说它带宽最优？

数据切成 $p$ 块，先做 $p-1$ 步 reduce-scatter，再做 $p-1$ 步 all-gather，每步每卡发 $N/p$。每卡总发送量为

$$2\frac{p-1}{p}N$$

$p$ 增大时趋于 $2N$，几乎与卡数无关，而任何 all-reduce 算法每卡至少要发送约 $\frac{p-1}{p}N$ 再收回同样多，所以 ring 在带宽项上是最优的。缺点是延迟项 $2(p-1)\alpha$ 随卡数线性增长，小消息时不划算。

## All-reduce、reduce-scatter、all-gather 三者是什么关系？

All-reduce = reduce-scatter + all-gather。reduce-scatter 之后每卡拿到求和结果的 $1/p$，all-gather 再把这些片段拼成完整结果。ZeRO-2/3、FSDP、Megatron SP 都是把 all-reduce 拆开，只保留自己那 $1/p$ 来省显存，总通信量不变。

## ZeRO-1/2/3 分别切了什么？每卡显存多少？

设参数量 $\Psi$、DP 度 $N_d$、混合精度 Adam（$K=12$）：

- ZeRO-1 切优化器状态：$4\Psi + 12\Psi/N_d$
- ZeRO-2 再切梯度：$2\Psi + 14\Psi/N_d$
- ZeRO-3 再切参数：$16\Psi/N_d$

ZeRO-1/2 通信量和 DDP 一样，ZeRO-3 是 1.5 倍。

## 70B 模型用 ZeRO-3 在 64 张卡上训练，每卡大概要多少显存？

模型状态 $16 \times 70\text{e}9 / 64 = 17.5$ GB。另外要算：

- 前向/反向时 all-gather 出来的整层参数：每层约 0.85B 参数，bf16 约 1.7 GB，开 prefetch 时同时有两层，约 3.4 GB；
- 激活：取决于序列长度和 micro-batch，开 activation checkpointing 后通常几 GB 到十几 GB；
- 通信 buffer、logits 和碎片。

总体在 80 GB 卡上可行。作为对比，ZeRO-1 每卡光参数和梯度就要 $4 \times 70 = 280$ GB。

## ZeRO-3 的通信量为什么是 DDP 的 1.5 倍？

DDP 的 all-reduce 约等于 $2\Psi$（reduce-scatter + all-gather）。ZeRO-3 平时只存参数分片：前向 all-gather 一次参数（$\Psi$），用完就丢；反向再 all-gather 一次（$\Psi$）；梯度 reduce-scatter 一次（$\Psi$）。合计 $3\Psi$，是 DDP 的 1.5 倍。

## FSDP 前向和反向分别发生什么通信？prefetch 做了什么？

以一个 wrap 单元（通常是一个 Transformer block）为单位：

- 前向：all-gather 该单元参数 → 计算 → 释放完整参数；
- 反向：再 all-gather 参数 → 计算梯度 → reduce-scatter 梯度到各自分片 → 释放。

prefetch 是在算第 $i$ 个单元时提前发起第 $i+1$ 个（反向时是前一个）单元的 all-gather，让通信藏在计算后面。

## FSDP1 和 FSDP2 有什么区别？

FSDP1 把一个 wrap 单元的所有参数拉平成一个一维 `FlatParameter` 再切分；FSDP2（`fully_shard`）对每个参数用 DTensor 沿第 0 维切。FSDP2 的好处：没有 FlatParameter 的拼接，显存更可预测；可以对单个参数做冻结、设不同精度；更容易和 TP 等其它 DTensor 并行组合。verl 里 `strategy=fsdp2` 即使用它。

## 什么是 HSDP？什么时候用？

Hybrid Sharded Data Parallel：在节点内做完全分片（FULL_SHARD），节点之间做复制，只在节点间 all-reduce 梯度。适合模型在一个节点（8 卡）内分片后就能放下、而跨节点带宽较慢的情况：参数 all-gather 这种高频通信只走 NVLink，跨节点只剩梯度同步。

## Megatron TP 里 MLP 为什么第一个线性层按列切、第二个按行切？

$Z=\text{GeLU}(XA)B$。$A$ 按列切后每卡得到 $Y_i=\text{GeLU}(XA_i)$，GeLU 是逐元素的，**不需要通信**；如果 $A$ 按行切，GeLU 之前就得先 all-reduce 求和。$B$ 按行切，每卡算 $Y_iB_i$ 得到部分和，最后一次 all-reduce 即可。这样整个 MLP 前向只需要 1 次 all-reduce。

## TP 每层要几次 all-reduce？为什么一般不跨节点？

每个 Transformer 层前向 2 次（attention 一次、MLP 一次），反向 2 次，每次通信 $b\cdot s\cdot h$ 个元素，而且都在关键路径上，没法和计算重叠。节点内 NVLink 每卡单向约 450 GB/s，跨节点 IB 每卡约 50 GB/s，跨节点做 TP 会被通信拖垮。所以 TP ≤ 8，放在节点内。

## Megatron 的 Sequence Parallel 解决什么问题？

TP 只切了 GEMM，LayerNorm、Dropout 的输入在 TP 组内是重复的完整激活。SP 把这些区域沿序列维切开，把原来的 all-reduce 换成 reduce-scatter + all-gather。通信量不变，这部分激活显存降到 $1/t$。注意它和 Ulysses / Ring Attention 不同，后者是把 attention 本身的序列维切开。

## Pipeline 的 bubble 怎么算？1F1B 比 GPipe 好在哪？

$p$ 个 stage、$m$ 个 micro-batch 时，bubble 占总时间 $\frac{p-1}{m+p-1}$（或者说是理想计算时间的 $\frac{p-1}{m}$ 倍）。1F1B 的 bubble 和 GPipe **一样大**，改进的是显存：GPipe 要存全部 $m$ 个 micro-batch 的激活，1F1B 前向几个之后就交替做反向，同时在飞的最多 $p$ 个，激活显存与 $m$ 无关。

## Interleaved 1F1B 是怎么减少 bubble 的？代价是什么？

每张卡持有 $v$ 段不相邻的层（虚拟 stage），一个 micro-batch 要在卡之间往返 $v$ 次，bubble 降为约 $\frac{p-1}{v\,m}$。代价是点对点通信次数变成 $v$ 倍，并且要求 micro-batch 数满足一定的整除关系。

## Ulysses 和 Ring Attention 都是切序列，有什么区别？

- Ulysses：attention 前用 all-to-all 把“按序列切”换成“按头切”，每卡拿完整序列、部分头，用任何现成 kernel 计算，再 all-to-all 换回来。并行度不能超过头数（GQA 时受 KV 头数约束）。
- Ring Attention / CP：K、V 块沿环传递，每卡用本地 Q 和收到的 KV 算部分结果，用 online softmax 合并，通信可以和计算重叠，不受头数限制；causal mask 下要用 zigzag 切分来均衡负载。

## MoE 的专家并行通信流程是什么？负载不均怎么办？

router 选 top-k 专家 → all-to-all **dispatch** 把 token 发到专家所在卡 → 专家 FFN → all-to-all **combine** 发回原卡 → 按门控权重加权求和。负载不均会让热门专家所在卡成为瓶颈，对策：auxiliary load-balancing loss、capacity factor + token drop、DeepSeek-V3 的 aux-loss-free 偏置调节、限制每个 token 路由到的节点数以控制跨节点流量。

## 混合精度 Adam 每参数 16 字节是怎么来的？

bf16 参数 2 + bf16 梯度 2 + fp32 主权重 4 + fp32 一阶动量 4 + fp32 二阶动量 4 = 16。很多框架把梯度累加在 fp32 里，会变成 18 字节甚至更多。所以 7B 全量微调光模型状态就是 112 GB。

## Activation checkpointing 的原理和代价？

前向只保存每层（或每个 checkpoint 段）的输入，反向时重新跑一遍这段前向得到中间激活。激活显存从“每层所有中间结果”降到“每层一个输入”，代价是多一次前向，训练计算从约 $6N$ 变成 $8N$，约多 33%。selective recomputation 只重算 attention 中“显存大、计算便宜”的部分，能用很小的计算代价省下大部分显存。

## 7B 模型训练 1T token，用 256 张 A100（bf16 312 TFLOPS），MFU 45%，大约多久？

总计算 $6 \times 7\text{e}9 \times 1\text{e}12 = 4.2\text{e}22$ FLOPs。集群有效算力 $256 \times 312\text{e}12 \times 0.45 \approx 3.6\text{e}16$ FLOPs/s。耗时约 $1.17\text{e}6$ 秒，约 13.5 天。

## MFU 怎么算？和 HFU 有什么区别？

$\text{MFU} = 6N \cdot \text{tokens/s} \,/\, (P_{\text{peak}} \times \#\text{GPU})$，分子只算模型“必须做”的计算。HFU 还把重计算等额外 FLOPs 计入，所以 HFU ≥ MFU。比较不同系统时用 MFU 更公平。长序列时分子还要加上 attention 项（每 token 约 $12\,L\,d\,T$）。

## 梯度累积和直接用大 batch 等价吗？

对“对所有样本取平均”的 loss，在正确归一化时梯度等价。但要注意两个坑：①如果每个 micro-batch 内先按自己的 token 数取平均，再在 micro-batch 之间平均，那么 token 数不同的 micro-batch 权重就不同，和全局 token-mean 不等价。verl 的 `agg_loss` 用全局的 `batch_num_tokens` 和 `dp_size` 归一化，就是为了让 loss 与 micro-batch 切法、并行度无关。②BatchNorm 之类依赖 batch 统计量的层不等价（LLM 一般用 LayerNorm/RMSNorm，没有这个问题）。

## bf16 训练还需要 loss scaling 吗？fp16 呢？

fp16 只有 5 位指数，最大 65504，小梯度容易下溢，需要动态 loss scaling：loss 乘以 $S$ 再反向，出现 inf/nan 就跳过这步并减小 $S$。bf16 有 8 位指数，范围和 fp32 一样，一般不需要 loss scaling；代价是尾数只有 7 位，所以累加要在 fp32 里做，主权重也保留 fp32。

## 多机训练卡住（hang），怎么排查？

最常见的原因是各 rank 调用集合通信的顺序或次数不一致：数据在某个 rank 上少一个 batch、条件分支不同、某些参数只在部分 rank 上有梯度。排查：开 `NCCL_DEBUG=INFO` 看卡在哪个通信；设置超时让它报错而不是一直挂起；用 py-spy dump 各 rank 的调用栈对比；检查各 rank 的 dataloader 长度；再排查网络（IB 端口、防火墙、`NCCL_SOCKET_IFNAME`）。

## 给 Llama-70B 在 128 张 H100（16 个节点）上做全量训练，你会怎么配并行？

方案一（简单）：FSDP 全分片 128 卡 + activation checkpointing。模型状态每卡 $16\times70/128 \approx 8.75$ GB，但每层参数的 all-gather 要跨节点，通信压力大，可以改用 HSDP 或加 TP。
方案二（Megatron 风格）：TP=8（节点内）× PP=4 × DP=4。每卡参数约 $70/32 \approx 2.2$B，参数加梯度约 8.75 GB，再用 distributed optimizer 把 12 字节/参数的优化器状态在 DP=4 上切开，约 6.6 GB，合计约 15 GB，剩下的显存给激活。PP=4 时 micro-batch 数至少取 16。
回答时说清取舍：TP 放节点内、PP 和 DP 跨节点、用 bubble 公式反推 micro-batch 数。

## 为什么 DDP 的梯度要分 bucket，而不是每个参数发一次或者最后一次性发？

每个参数单独发：消息太小，延迟项 $\alpha$ 占主导，launch 次数多。全部算完再一次性发：没法和反向计算重叠，通信全部暴露。bucket（默认 25 MB）按反向产生梯度的顺序（参数逆序）组织，凑满一个就异步 all-reduce，在消息大小和重叠程度之间取平衡。梯度累积时用 `no_sync()` 跳过中间步骤的同步。
