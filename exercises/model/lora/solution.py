import torch


class LoRALinear:
    def __init__(self, W, r, alpha, generator):
        self.W = W.detach().clone()  # 复制并切断计算图：基座权重冻结，也不会改到调用者的 W
        out_f, in_f = self.W.shape
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        A = torch.randn(r, in_f, generator=generator, dtype=self.W.dtype) / r  # 标准差 1/r
        self.A = A.to(self.W.device).requires_grad_(True)
        self.B = torch.zeros(out_f, r, dtype=self.W.dtype, device=self.W.device, requires_grad=True)
        self.merged = False

    def forward(self, x):
        y = x @ self.W.T
        if not self.merged:
            y = y + self.scaling * (x @ self.A.T) @ self.B.T
        return y

    @torch.no_grad()  # 否则 W 会挂上 A、B 的计算图
    def merge(self):
        if not self.merged:
            self.W = self.W + self.scaling * self.B @ self.A
            self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.W = self.W - self.scaling * self.B @ self.A
            self.merged = False

    def num_trainable_params(self):
        return self.A.numel() + self.B.numel()
