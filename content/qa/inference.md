---
title: 推理系统
summary: KV cache、调度、前缀复用、PD 分离、投机解码与推理估算题。
tags: Infra, 推理, vLLM
order: 40
---

## Prefill 和 decode 的瓶颈分别是什么？

Prefill 一次处理整段 prompt，每个权重被 $T$ 个 token 复用，算术强度很高，是**算力受限**，决定 TTFT。Decode 每步每条序列只有 1 个新 token，权重的复用度约等于 batch 大小，attention 部分还要读整个 KV cache，是**访存受限**，决定 TPOT。所以 prefill 优化看 FLOPs 利用率，decode 优化看搬了多少字节。

## 7B 模型 bf16，单张 H100 单条请求的 decode 速度上限大约多少？

每生成一个 token 至少读一遍权重 14 GB，HBM 带宽约 3.35 TB/s，上限约 $3.35\text{e}12 / 14\text{e}9 \approx 240$ token/s。若上下文 4K、MHA 的 KV 约 2 GB，每步读 16 GB，约 210 token/s。实际还要打折（kernel 跑不满带宽、launch 开销）。

## KV cache 大小怎么算？Llama-2-70B 每个 token 多少？

$$2 \times L \times H_{kv} \times d_{\text{head}} \times \text{bytes} \times \text{tokens}$$

Llama-2-70B：80 层、8 个 KV 头（GQA）、head dim 128、bf16，每 token $2\times80\times8\times128\times2 = 320$ KB。32 条 4K 上下文的请求就要约 40 GB。

## PagedAttention 解决了什么问题？block size 怎么取？

解决 KV cache 的碎片和无法共享：按固定大小的 block 分配，block table 记录逻辑到物理的映射，只有最后一块可能没填满，浪费从 60%–80% 降到 4% 以内，还能用引用计数在序列之间共享前缀块。block size 太小：block table 变长、kernel 访存不连续、管理开销大；太大：最后一块的内部碎片变多、前缀共享的粒度变粗。vLLM 默认 16。

## Copy-on-write 用在什么场景？

多个序列共享同一段前缀 KV 时：并行采样（$n>1$，比如 GRPO 一个 prompt 采 8 条）、beam search、前缀缓存命中。共享块引用计数大于 1，某个序列要往共享块里写新 token 时才复制一份私有块，其余序列不受影响。

## Continuous batching 和 static batching 的区别？

static batching 凑齐一批一起跑，要等最长的那条生成完才换下一批，短请求的槽位空转。continuous batching 每个 decode 步之后都重新调度（iteration-level scheduling）：结束的请求立刻退出，新请求立刻加入。配合分页的 KV cache，batch 始终是满的，吞吐可以高出数倍。

## Chunked prefill 解决什么问题？代价是什么？

长 prompt 的整段 prefill 会占住 GPU 几百毫秒，期间正在 decode 的请求全部停顿，TPOT 出现尖刺。chunked prefill 把长 prefill 切块，每步在 token 预算内混合“一块 prefill + 所有 decode token”。好处是 decode 不再被阻塞，且算力受限的 prefill 和访存受限的 decode 互补；代价是长 prompt 的 TTFT 略有增加。

## vLLM 的 prefix caching 和 SGLang 的 RadixAttention 有什么区别？

vLLM 对每个满 block 计算哈希（包含父块哈希和本块 token），哈希相同就复用物理块，引用计数为 0 的块按 LRU 淘汰。SGLang 用 radix tree 组织所有缓存的 token 序列，新请求做最长前缀匹配，按 LRU 淘汰叶子，并在调度时优先处理前缀匹配更长的请求（cache-aware scheduling）。前者粒度是 block，后者是 token 级的树结构，两者目标一致。

## PD 分离的好处和代价？什么情况下不值得做？

好处：prefill 和 decode 不再互相干扰；TTFT 和 TPOT 可以分别优化；两个池可以用不同的并行配置、独立扩缩容。代价：KV cache 要跨机传输，长 prompt 的 KV 很大，吃带宽；系统复杂度高。如果 prompt 普遍很短、规模不大，或者网络带宽不足，chunked prefill 往往就够了，PD 分离收益有限。

## CUDA Graph 为什么能加速 decode？有什么限制？

decode 每步有几百个小 kernel，小 batch 时 CPU launch kernel 的时间可能超过 GPU 计算时间。CUDA Graph 把一整步录制下来一次回放，几乎消除 launch 开销。限制：shape 和显存地址固定，所以要按若干 batch 大小分别录制、运行时 pad 到最近的大小；录制要额外时间和显存；动态控制流和变长 shape（如 prefill）难以直接录制。

## TTFT、TPOT、吞吐、goodput 分别是什么？

TTFT：请求到达到第一个 token；TPOT：后续每个 token 的平均间隔；E2E 延迟 = TTFT + TPOT ×（输出长度 − 1）；吞吐：每秒生成的 token 数或完成的请求数；goodput：同时满足 SLO（例如 TTFT < 2s、TPOT < 50ms）的请求速率。只看吞吐会掩盖延迟问题，线上更关心 goodput。

## KV 不够时抢占请求，swap 和 recompute 怎么选？

swap 把被抢占请求的 KV 搬到 CPU 内存，恢复时搬回，代价是 PCIe 带宽；recompute 直接丢掉 KV，恢复时把“prompt + 已生成部分”重新 prefill 一遍，代价是计算。prefill 是算力受限且很快，而 PCIe 带宽有限，所以多数情况下 recompute 更划算（vLLM V1 默认用 recompute，待核实）。

## GQA、MQA、MLA 对推理有什么影响？

它们都在减少每 token 的 KV 大小：MHA 每层每 token 缓存 $2n_hd_h$ 个元素，GQA 降为 $2gd_h$，MQA 降为 $2d_h$，MLA 只缓存一个压缩 latent 加一小段 RoPE key（DeepSeek-V3 为 576）。KV 越小，同样显存能放越多并发请求，decode 每步读的字节也越少；GQA 还让每份 K、V 被多个 query 头复用，提高了 attention 的算术强度。

## FlashAttention 为什么快？

它没有减少 FLOPs（反向甚至更多），而是减少了 HBM 读写：把 Q、K、V 分块搬进 SRAM，用 online softmax（维护每行最大值和指数和）在片上算完，不把 $N\times N$ 的 score 矩阵写回 HBM。HBM 访问从 $\Theta(Nd+N^2)$ 降到 $\Theta(N^2d^2/M)$，显存从 $O(N^2)$ 降到 $O(N)$。

## Flash-Decoding 解决的是什么问题？

decode 时每个请求的 query 只有 1 行，如果 batch × 头数不够多，按 batch 和头并行填不满所有 SM，而长上下文的 KV 又很长。Flash-Decoding 沿 KV 长度再切分，多个 block 并行算各自那段的部分结果，最后用 logsumexp 合并，提升长上下文、小 batch 下的 decode 速度。

## 投机解码为什么是无损的？

草稿 $x\sim q$，以 $\min(1,p(x)/q(x))$ 接受，拒绝则从 $\text{norm}(\max(0,p-q))$ 重采样。输出 $x$ 的概率 = 被接受的部分 $\min(p(x),q(x))$ + 被拒后重采到 $x$ 的部分 $\max(0,p(x)-q(x))$ = $p(x)$。拒绝概率 $1-\sum\min(p,q)$ 恰好等于修正分布的归一化常数，所以两项可以直接相加。

## 投机解码的期望接受长度和加速比怎么算？

每个位置独立以概率 $\alpha$ 接受、草稿长度 $\gamma$ 时，每轮产出 token 数的期望为 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$（含 bonus/修正 token）。草稿单步耗时是目标模型的 $c$ 倍时，加速比 $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$。例如 $\alpha=0.8$、$\gamma=4$、$c=0.05$，约 2.8 倍。

## 为什么高并发时投机解码可能没收益甚至变慢？

收益的前提是 decode 访存受限、多验证几个 token 几乎免费。batch 很大时目标模型接近算力受限，每多验证一个 token 就多一份计算，还挤掉了别的请求的 batch 容量；被拒的 token 是白算；草稿模型本身也有开销；可变的接受长度让 CUDA Graph 和 batching 更难。所以需要按负载动态调整验证长度（DSpark 的置信度调度就是在解决这个问题），高负载时甚至直接关掉。

## Medusa、EAGLE、MTP 的草稿机制有什么不同？

Medusa 在目标模型最后一层上接多个独立的头，第 $k$ 个头直接预测第 $k$ 个后续 token，各头之间互不知道对方猜了什么。EAGLE 用一层轻量 Transformer 在特征层自回归，输入目标模型的隐藏特征和下一个 token 的 embedding；EAGLE-3 改为直接预测 token、融合多层特征。MTP（DeepSeek-V3）是预训练时就一起训练的多 token 预测模块，推理时直接当草稿用，报告第二个 token 接受率 85%–90%。

## W4A16 和 W8A8 分别加速哪个阶段？

W4A16（GPTQ、AWQ）只量化权重，计算时反量化回 16 位，减少的是读权重的字节数，主要加速**访存受限的 decode**（小 batch 时尤其明显）。W8A8（SmoothQuant 的 INT8、Hopper 上的 FP8）权重和激活都低精度，能用低精度 Tensor Core 做 GEMM，主要加速**算力受限的 prefill** 和大 batch。KV cache 量化则减少 KV 显存，提升可并发数。

## 一张 80 GB 卡跑 Llama-3-8B（bf16），大概能并发多少条 4K 上下文的请求？

权重约 16 GB。每 token KV $= 2\times32\times8\times128\times2$ B = 128 KB（32 层、8 个 KV 头），一条 4K 请求约 0.5 GB。留约 10 GB 给激活和余量，剩约 54 GB 给 KV，约 100 条。先写公式再代数，并说明这是上限，实际还受 `max_num_seqs`、碎片和调度策略影响。

## Top-p（nucleus）采样怎么实现？temperature 起什么作用？

temperature：logits 除以 $T$ 再 softmax，$T<1$ 让分布更尖、$T>1$ 更平，$T\to0$ 退化为贪心。top-p：把概率从大到小排序，取累计概率首次达到 $p$ 的最小前缀集合，集合外概率置零，再重新归一化后采样。top-k 则只保留概率最大的 $k$ 个。RL rollout 通常用 $T=1$ 且不截断，保证采样分布就是策略本身，否则 log-prob 和重要性比都要相应修正。

## RL 的 rollout 推理和线上服务推理有什么不同？

①目标是整批吞吐而不是单请求延迟，所有 prompt 一次性提交；②每个 step 权重都会更新，前缀缓存和 CUDA Graph 等都要考虑失效与重建，引擎要支持 sleep/wake 和快速换权重；③需要返回每个 token 的 log-prob；④长尾非常明显，最后少数超长请求拖住整个 step；⑤和训练框架共享 GPU 时显存要精打细算（如 `gpu_memory_utilization=0.5`）；⑥要尽量和训练侧数值一致。

## 为什么推理引擎算出的 log-prob 和训练框架算的不一致？有什么影响？

两边的 kernel 实现（attention、matmul、归约顺序）、并行方式（TP 切法）、batch 组成都不同，浮点运算不满足结合律，bf16 下差异更明显；MoE 还可能因数值差异选到不同专家。影响是：用推理侧采样、训练侧算概率时，数据其实是“轻微 off-policy”的，重要性比偏离 1，可能导致训练不稳定。对策是记录 rollout 的 log-prob，用 TIS 之类的截断重要性采样修正，或者提升两侧的数值一致性。

## 多个推理副本之间怎么做负载均衡？

基本做法是按当前负载（排队请求数、KV 占用）路由到最空闲的副本。进阶：前缀感知路由，把前缀相同的请求发到同一副本以提高 prefix cache 命中率；多轮会话用 sticky session，同一会话始终发往同一副本（verl agent loop 的 server client 就是首轮按最少负载选副本、之后按 request_id 粘住）。PD 分离时还要分别均衡 prefill 池和 decode 池。
