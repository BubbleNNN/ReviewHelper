---
title: LoRA 线性层（前向、合并与拆分）
difficulty: 易
tags: 微调, LoRA, PEFT
order: 34
source: Hu et al. 2021 LoRA；HF peft 的 LoraLayer.merge / unmerge
---

实现一个带 LoRA 旁路的线性层。基座权重 `W` 形状 `(out, in)`（PyTorch `nn.Linear` 约定），输入 `x` 形状 `(..., in)`：

$$y = xW^\top + \frac{\alpha}{r}\,x A^\top B^\top,\qquad A\in\mathbb R^{r\times in},\ B\in\mathbb R^{out\times r}$$

```python
class LoRALinear:
    def __init__(self, W, r, alpha, rng): ...
    W, A, B, r, alpha, scaling, merged      # 属性名固定，测试会直接读写 A、B
    def forward(self, x)                    # (..., in) -> (..., out)
    def merge(self)                         # 把 scaling * B @ A 加进 W，之后 forward 只用 W
    def unmerge(self)                       # 撤销 merge
    def num_trainable_params(self) -> int
```

- 构造时**复制** `W`（不能修改调用者传入的数组）。
- 初始化：`A` 用 `rng.normal(0, 1/r, size=(r, in))`，**`B` 全零**；`scaling = alpha / r`，`merged = False`。
- `merge()` 重复调用不能重复叠加；`unmerge()` 在未合并时什么也不做。
- 可训练参数只有 `A` 和 `B`。

<!-- solution -->

### 讲解

**为什么 B 初始化为 0**：这样训练开始时 $\Delta W = BA = 0$，模型输出和基座完全一致，微调从预训练权重平滑出发。A 不能也是 0，否则 $A$、$B$ 的梯度都为 0（$\partial/\partial B \propto xA^\top=0$，$\partial/\partial A\propto B^\top(\cdot)=0$），永远学不动。

**参数量**：全量微调一个 $d\times d$ 的矩阵要 $d^2$ 个参数，LoRA 只要 $2rd$。$d=4096,\ r=16$ 时约是 1/128。优化器状态（Adam 的 m、v）也随之大幅减少，这是 LoRA 省显存的主要来源。

**为什么乘 $\alpha/r$**：调 $r$ 时不用重新调学习率，$\alpha$ 控制更新幅度。（rsLoRA 主张用 $\alpha/\sqrt r$，在大 $r$ 时更稳定。）

**merge 的意义**：部署时把 $\frac{\alpha}{r}BA$ 加进 $W$，推理没有额外延迟。多 LoRA 服务（如 S-LoRA / vLLM multi-LoRA）则不合并，同一个 batch 里不同请求挂不同的 adapter。

**和 RL 的联系**：verl 支持 LoRA RL。actor 用 LoRA 时，参考模型就是“关掉 adapter 的同一个模型”，不需要单独加载 ref 权重；同步给 rollout 引擎时可以只传 adapter 权重。

> [!Q]
> LoRA 应该加在哪些层？r 取多大？
> 原论文只加在 attention 的 $W_q, W_v$ 上；后续实践（如 QLoRA）发现加到所有线性层（包括 MLP）效果更好。r 常用 8–64，任务越难、和预训练分布差得越远，需要的秩越高。RL 场景下有工作报告很小的 r 也够用，但具体数值待核实。

```python
# torch
class LoRALinear(nn.Module):
    def forward(self, x):
        return F.linear(x, self.W) + self.scaling * F.linear(F.linear(x, self.A), self.B)
```
