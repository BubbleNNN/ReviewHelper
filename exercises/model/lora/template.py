import torch


class LoRALinear:
    """y = x W^T + (alpha / r) * x A^T B^T

    属性：W (out, in)、A (r, in)、B (out, r) 都是 tensor；r (int)、alpha、scaling (python float)、merged (bool)
    """

    def __init__(self, W, r, alpha, generator):
        """初始化。

        Args:
            W: (out, in) float tensor，基座权重（可能带 requires_grad=True，比如来自 nn.Linear.weight）
            r: python int，秩
            alpha: python float
            generator: torch.Generator，A 的随机数只能来自它
        要求：
            self.W：W 的副本，detach 掉、requires_grad=False（冻结，且不能改到调用者传入的 W）
            self.A：torch.randn(r, in, generator=generator, dtype=W.dtype) / r（标准差 1/r），requires_grad=True
            self.B：(out, r) 全零，dtype 与 W 相同，requires_grad=True
            self.scaling = alpha / r，self.merged = False
        """
        raise NotImplementedError

    def forward(self, x):
        """x: (..., in) float tensor -> (..., out) tensor，dtype 与 x 相同。merged=True 时只用 W。"""
        raise NotImplementedError

    def merge(self):
        """W += scaling * B @ A；重复调用不重复叠加。在 torch.no_grad() 下改 W，合并后 W 仍然 requires_grad=False。"""
        raise NotImplementedError

    def unmerge(self):
        """撤销 merge；未合并时不做事。同样在 torch.no_grad() 下改 W。"""
        raise NotImplementedError

    def num_trainable_params(self):
        """A 与 B 的元素个数之和，返回 python int。"""
        raise NotImplementedError
