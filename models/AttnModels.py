#!/usr/bin/env python3
# encoding: utf-8

import torch
from torch import nn
import math


class SelfAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_head: int,
        scaling: int,
        seqlen_scaling: bool = True,
        attn_bias: bool = False,
        pair_feat_dim: int = -1,
        pair_feat_norm: bool = False,
        gate_on_V: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.n_head = n_head
        self.scaling = scaling
        self.seqlen_scaling = seqlen_scaling
        self.attn_bias = attn_bias
        self.gate_on_V = gate_on_V

        self.to_q = nn.Linear(in_dim, out_dim, bias=False)
        self.to_k = nn.Linear(in_dim, out_dim, bias=False)
        self.to_v = nn.Linear(in_dim, out_dim, bias=False)

        if self.attn_bias:
            self.to_attn_bias = nn.Sequential(
                nn.LayerNorm(pair_feat_dim) if pair_feat_norm else nn.Identity(),
                nn.Linear(pair_feat_dim, n_head, bias=False),
            )

        if self.gate_on_V:
            self.gate_layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Sigmoid(),
            )

    def forward(self, x, pair_feat, res_mask):
        # x: [n_batch, n_row, n_col, n_head*n_emb]
        n_batch, n_row, n_col, n_head_x_n_emb = x.size()
        scaling = (
            self.scaling / math.sqrt(n_col) if self.seqlen_scaling else self.scaling
        )

        # Q, K, V: [n_batch, n_row, n_col, n_head, n_emb]
        Q = self.to_q(x).view(n_batch, n_row, n_col, self.n_head, -1)
        K = self.to_k(x).view(n_batch, n_row, n_col, self.n_head, -1)
        V = self.to_v(x).view(n_batch, n_row, n_col, self.n_head, -1)

        Q = Q * scaling

        QK = torch.einsum(f"nrihd,nrjhd->nhrij", Q, K)

        if self.attn_bias:
            _attn_bias = (
                self.to_attn_bias(pair_feat)
                .permute(0, 3, 1, 2)
                .view(n_batch, self.n_head, 1, n_col, n_col)
            )
            QK = QK + _attn_bias

        if res_mask is not None:
            res_mask = res_mask[:, :, None] * res_mask[:, None, :]
            QK.masked_fill_(
                res_mask[:, None, None, :, :] == 0, torch.finfo(QK.dtype).min
            )

        attn = QK.softmax(-1)

        out = torch.einsum(f"nhrij,nrjhd->nrihd", attn, V)

        if self.gate_on_V:
            gate = self.gate_layer(x).view(n_batch, n_row, n_col, self.n_head, -1)
            out = gate * out

        out = out.contiguous().view(n_batch, n_row, n_col, -1)

        return out


class MHSelfAttention(nn.Module):
    """
    Multihead attention on row (i.e. residues in each sequence), with each head being a Fast Attention Head.
    Agrs:
        emb_dim (int): the embedding dim.
        n_head (int): the head num.
        head_by_head (bool): True: calc each head one after another, False: calc all heads simultaneously.
    """

    def __init__(
        self,
        emb_dim: int,
        n_head: int,
        attn_config: dict = None,
        seqlen_scaling: bool = False,
        head_by_head: bool = False,
    ):
        super().__init__()
        self.n_head = n_head
        self.head_dim = emb_dim // n_head
        self.seqlen_scaling = seqlen_scaling
        self.head_by_head = head_by_head
        self.scaling = self.head_dim**-0.5

        # calc each head one after another
        if self.head_by_head:
            self.heads = nn.ModuleList()
            for i in range(n_head):
                self.heads.append(
                    self.get_attn_module(emb_dim, self.head_dim, 1, attn_config)
                )
        # calc all heads simultaneously
        else:
            self.heads = self.get_attn_module(emb_dim, emb_dim, n_head, attn_config)

        self.to_out = nn.Linear(emb_dim, emb_dim)

    def get_attn_module(self, in_dim, out_dim, n_head, attn_config):
        return SelfAttention(
            in_dim,
            out_dim,
            n_head,
            scaling=self.scaling,
            seqlen_scaling=self.seqlen_scaling,
            **attn_config,
        )

    def forward(self, x, pair_feat, res_mask):
        # calc each head one after another
        if self.head_by_head:
            out = []
            for i in range(self.n_head):
                head_out = self.heads[i](x, pair_feat, res_mask)
                out.append(head_out)
            # concat
            out = torch.cat(out, dim=-1)
        else:
            out = self.heads(x, pair_feat, res_mask)

        out = self.to_out(out)

        return out


class TriangleMultiplication(nn.Module):
    """
    Triangle Multiplication.
    """

    def __init__(self, emb_dim: int, op_type: str, *args, **kwargs):
        super().__init__()

        self.op_type = op_type

        if op_type == "outgoing":
            self.einsum_eq = "bikc,bjkc->bijc"
        elif op_type == "incoming":
            self.einsum_eq = "bkjc,bkic->bijc"

        self.left_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim)
        )
        self.right_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim)
        )

        self.left_gate = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Sigmoid())
        self.right_gate = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Sigmoid())

        self.out_gate = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Sigmoid())

        self.out_layer = nn.Sequential(
            nn.LayerNorm(emb_dim), nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, pair_feat, res_mask=None, *args, **kwargs):
        # pair_feat: [n_batch, L, L, n_emb]

        proj_left = self.left_proj(pair_feat)
        proj_right = self.right_proj(pair_feat)

        # res_mask
        if res_mask is not None:
            res_mask = (res_mask[:, :, None] * res_mask[:, None, :]).unsqueeze(-1)
            proj_left = proj_left * res_mask
            proj_right = proj_right * res_mask

        gate_left = self.left_gate(pair_feat)
        gate_right = self.right_gate(pair_feat)

        proj_left = proj_left * gate_left
        proj_right = proj_right * gate_right

        out = torch.einsum(self.einsum_eq, proj_left, proj_right)

        out = self.out_layer(out)

        gate = self.out_gate(pair_feat)
        out = out * gate

        return out
