import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AdditionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        nhead: int = 8,
        layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        # MPS 暂不支持 encoder fast path 用到的 _nested_tensor_from_mask_left_aligned，
        # 关闭 nested tensor 与 mask 检查以保持 CUDA/MPS/CPU 行为一致
        # （不同 PyTorch 版本属性名不同，两个都设以兼容）
        self.transformer.encoder.enable_nested_tensor = False
        self.transformer.encoder.use_nested_tensor = False
        self.transformer.encoder.mask_check = False
        self.output = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = src.eq(self.pad_id)
        tgt_key_padding_mask = tgt.eq(self.pad_id)
        tgt_mask = self.make_causal_mask(tgt.size(1), tgt.device)
        src_emb = self.position(self.embedding(src) * self.scale)
        tgt_emb = self.position(self.embedding(tgt) * self.scale)
        hidden = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output(hidden)

    @staticmethod
    def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
