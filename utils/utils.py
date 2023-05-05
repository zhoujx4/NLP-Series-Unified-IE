"""
@Time : 2023/4/2411:13
@Auth : zhoujx
@File ：utils.py
@DESCRIPTION:

"""
import random

import numpy as np
import torch


def set_seeds(seed=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 1
        self.total = 1

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class RunningEMA():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self, decay=0.995):
        self.avg = 1
        self.decay = decay

    def update(self, val):
        self.avg = (1.0 - self.decay) * val + self.decay * self.avg

    def __call__(self):
        return self.avg


def scatter_nd_pytorch(indices, updates, shape):
    # 创建一个形状为 shape 的全零张量
    result = torch.zeros(shape, dtype=updates.dtype, device=updates.device)

    # 使用 index_put_ 更新张量的值
    result.index_put_(tuple(indices.t()), updates, accumulate=False)

    return result


def decode():
