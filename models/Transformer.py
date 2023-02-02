#!/usr/bin/env python3
# encoding: utf-8

from torch import nn
from .AttnModels import MHSelfAttention
from .Layer import FeedForward, Residual


class TransformerLayer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_head: int,
        attn_config: dict = None,
        bias: bool = False,
        seqlen_scaling: bool = True,
        n_ff_hidden: int = None,
        activation: str = "elu",
        head_by_head: bool = True,
    ):
        super().__init__()

        self.head_dim = emb_dim // n_head

        attn_row = MHSelfAttention(
            emb_dim, n_head, attn_config, seqlen_scaling, head_by_head
        )
        self.attn_row = Residual(attn_row, emb_dim)

        ff_layer = FeedForward(emb_dim, emb_dim, n_ff_hidden, activation, bias)
        self.ff_layer = Residual(ff_layer, emb_dim)

    def forward(self, x, pair_feat, res_mask):

        x = self.attn_row(x, pair_feat, res_mask)

        x = self.ff_layer(x)

        return x
