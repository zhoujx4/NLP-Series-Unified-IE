"""
@Time : 2023/4/2620:40
@Auth : zhoujx
@File ：layers.py.py
@DESCRIPTION:

"""
import torch
from torch import nn

INF = 1e12


class EfficientGlobalPointer(nn.Module):
    def __init__(
            self,
            hidden_size,
            heads=12,
            head_size=64,
            RoPE=True,
            use_bias=True,
            tril_mask=True,
            max_length=512, ):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense1 = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.dense2 = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if RoPE:
            self.rotary = RotaryPositionEmbedding(head_size, max_length)

    def forward(self, inputs, attention_mask=None):
        inputs = self.dense1(inputs)  # [B, L, head_size * 2]
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE编码
        if self.RoPE:
            qw, kw = self.rotary(qw), self.rotary(kw)

        # 计算内积
        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / self.head_size ** 0.5  # [B, L, L]
        bias = self.dense2(inputs).transpose(1, 2) / 2  # 'bnh->bhn'  [B, head * 2, L]
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # [B, N, L, L]

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (1 - attention_mask[:, None, None, :] *
                         attention_mask[:, None, :, None])
            logits = logits - attn_mask * INF

        # 排除下三角
        if self.tril_mask:
            # 排除下三角
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)

            logits = logits - mask * INF

        return logits


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_length=512, head_axis=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_length, dtype=inv_freq.dtype)
        freqs = torch.einsum("n , d -> n d", t, inv_freq)
        if head_axis is not None:
            freqs = freqs.unsqueeze(head_axis)

        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.register_buffer("cos", freqs.cos(), persistent=False)

    def forward(self, t, offset=0):
        # t shape [bs, seqlen, h, dim]
        seqlen = t.shape[1]
        sin, cos = (
            self.sin[offset:offset + seqlen, :],
            self.cos[offset:offset + seqlen, :],)

        t1, t2 = t[..., 0::2], t[..., 1::2]
        # 奇偶交错
        return torch.stack(
            [t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1).flatten(-2, -1)