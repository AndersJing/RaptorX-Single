#!/usr/bin/env python3
# encoding: utf-8

import torch
from torch import nn

from .Transformer import TransformerLayer
from .Layer import FeedForward, Residual
from .AttnModels import TriangleMultiplication, MHSelfAttention


class CovLayer(nn.Module):
    """
    Covariance layer.
    """

    def __init__(self, n_emb_1D=32, n_emb_2D=128, *args, **kwargs):
        super().__init__()

        self.proj_layer = nn.Linear(n_emb_1D**2, n_emb_2D)

    def calc_pair_outer_product(self, x_left, x_right):
        n_batch, seq_num, n_res, emb_dim = x_left.shape

        outer_product = torch.einsum("bnil,bnjr->bijlr", x_left, x_right)
        outer_product = outer_product.reshape(n_batch, n_res, n_res, -1)

        return outer_product

    def forward(self, x_left, x_right, *args, **kwargs):
        outer_product = self.calc_pair_outer_product(x_left, x_right)
        pair_feat = self.proj_layer(outer_product)

        return pair_feat


class seq_to_pair(nn.Module):
    """
    Generate pairwise feature from seq feature.
    """

    def __init__(self, emb_dim=256, n_emb_1D=32, n_emb_2D=128):
        super().__init__()
        self.n_emb_2D = n_emb_2D

        self.layer_norm_seq = nn.LayerNorm(emb_dim)

        # 1D feat
        self.proj_1D_left = nn.Sequential(
            nn.Linear(emb_dim, n_emb_1D),
            nn.LayerNorm(n_emb_1D),
        )

        self.proj_1D_right = nn.Sequential(
            nn.Linear(emb_dim, n_emb_1D),
            nn.LayerNorm(n_emb_1D),
        )

        # outer product
        self.cov_layer = CovLayer(n_emb_1D, n_emb_2D)

    def get_out_channel(self):
        return self.n_emb_2D

    def get_pair_feat_1D(self, x_left, x_right):
        seq_len = x_left.shape[-2]

        x_left = x_left.unsqueeze(1).repeat(1, seq_len, 1, 1)
        x_right = x_right.unsqueeze(2).repeat(1, 1, seq_len, 1)

        return torch.cat((x_left, x_right), dim=-1)

    def forward(self, x, pair_feat_prev=None, *args, **kwargs):
        # x: [n_batch, seq_num, seq_len, emb_dim]

        x = self.layer_norm_seq(x)

        # 1D project
        feat_1D_left = self.proj_1D_left(x)
        feat_1D_right = self.proj_1D_right(x)

        # outer product
        pair_feat = self.cov_layer(feat_1D_left, feat_1D_right, *args, **kwargs)

        pair_feat = pair_feat + pair_feat_prev

        # n_batch, seq_len, seq_len, n_emb
        return pair_feat


class PairUpdateLayer(nn.Module):
    """
    The main Module to update pairwise feature.
    """

    def __init__(
        self,
        emb_dim: int,
        TriangleAttn_config: dict = None,
        activation: str = "relu",
    ):
        super().__init__()

        # seq to pair
        self.seq_to_pair_layer = seq_to_pair()

        # Triangle Multiplication
        self.triangle_mul_outgoing = Residual(
            TriangleMultiplication(emb_dim=emb_dim, op_type="outgoing"), emb_dim
        )
        self.triangle_mul_incoming = Residual(
            TriangleMultiplication(emb_dim=emb_dim, op_type="incoming"), emb_dim
        )

        # Triangle Attention
        self.triangle_attn_starting = Residual(
            MHSelfAttention(emb_dim, **TriangleAttn_config), emb_dim
        )
        self.triangle_attn_ending = Residual(
            MHSelfAttention(emb_dim, **TriangleAttn_config), emb_dim
        )

        # FeedForward
        ff_layer = FeedForward(
            emb_dim, emb_dim, n_hidden=emb_dim * 4, activation=activation
        )
        self.ff_layer = Residual(ff_layer, emb_dim)

    def forward(self, x, pair_feat, res_mask, *args, **kwargs):

        # pair feature
        pair_feat = self.seq_to_pair_layer(x, pair_feat, **kwargs)

        # Triangle Multiplication
        pair_feat = self.triangle_mul_outgoing(pair_feat, res_mask, **kwargs)
        pair_feat = self.triangle_mul_incoming(pair_feat, res_mask, **kwargs)

        # Triangle Attn
        pair_feat = self.triangle_attn_starting(
            pair_feat, pair_feat, res_mask, **kwargs
        )

        pair_feat = pair_feat.transpose(1, 2)
        pair_feat = self.triangle_attn_ending(pair_feat, pair_feat, res_mask, **kwargs)
        pair_feat = pair_feat.transpose(1, 2)

        # feed forward
        pair_feat = self.ff_layer(pair_feat, *args, **kwargs)

        return pair_feat


class EvoformerLayer(nn.Module):
    """
    Hybrid Attention layer: TransformerLayer and PairUpdate layer.
    """

    def __init__(self, emb_dim, seq_config, pair_config):
        super().__init__()

        self.transformer_layer = TransformerLayer(**seq_config, emb_dim=emb_dim)
        self.pair_update_layer = PairUpdateLayer(**pair_config)

    def forward(self, x, pair_feat, res_mask):
        x = self.transformer_layer(x, pair_feat, res_mask)
        pair_feat = self.pair_update_layer(x, pair_feat, res_mask)
        return x, pair_feat


class Evoformer(nn.Module):
    def __init__(
        self,
        n_input: int = 256,
        n_layer: int = 24,
        seq_config: dict = {},
        pair_config: dict = {},
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(EvoformerLayer(n_input, seq_config, pair_config))

    def forward(self, x, pair_feat, res_mask):
        """
        x shape: [n_batch, seq_num, seq_len, n_emb]
        """

        # Evoformer layers
        for i, layer in enumerate(self.layers):
            x, pair_feat = layer(x, pair_feat, res_mask)

        return x, pair_feat
