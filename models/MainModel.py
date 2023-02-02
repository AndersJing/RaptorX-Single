#!/usr/bin/env python3
# encoding: utf-8

import torch
from torch import nn

import utils.Utils as Utils

from .Embedding import EmbeddingModule
from .Evoformer import Evoformer
from .PairModule import PairModule
from .StrucModule import StrucModule


class MainModel(nn.Module):
    def __init__(self, configs, *args, **kwargs):
        super().__init__()

        self.n_alphabet = configs["n_alphabet"]

        # embedding module
        self.emb_module = EmbeddingModule(
            plm_config=configs["EMBEDDING_MODULE"]["plm_config"],
            struc_dist_bins=configs["EMBEDDING_MODULE"]["struc_dist_bins"],
        )

        # seq module
        self.seq_module = Evoformer(**configs["SEQ_MODULE"])

        # pair module
        n_pair_dim = configs["SEQ_MODULE"]["pair_config"]["emb_dim"]
        self.pair_module = PairModule(n_pair_dim, configs["PAIR_MODULE"])

        # structure module
        self.structure_model = StrucModule()
        self.struc_out_keys = configs["STRUC_MODULE"]["out_keys"]

    def forward(self, data, *args, **kwargs):
        seq_encoding = data["seq_encoding"]  # (n_batch, seq_len)
        ESM_token = data["ESM_token"]  # (n_batch, seq_len, C) or (n_batch, seq_len)
        ProtTrans_token = data["ProtTrans_token"]  # (n_batch, seq_len+1)
        recycling = data["recycling"]
        recycle_data = data["recycle_data"]

        # residue mask
        res_mask = torch.where(seq_encoding < self.n_alphabet, 1.0, 0.0)

        # embedding module
        seq_feat, pair_feat, plm_feat = self.emb_module(
            seq_encoding, ESM_token, ProtTrans_token, recycle_data, res_mask
        )

        # seq module
        seq_feat, pair_feat = self.seq_module(seq_feat, pair_feat, res_mask)
        query_feat = seq_feat[:, 0, :, :]

        # pair module
        pair_out, pair_feat = self.pair_module(pair_feat)

        # structure module
        struc_out = self.structure_model(query_feat, pair_feat, res_mask)

        outs = pair_out + struc_out

        # recycle data
        recycle_data = {}
        if recycling:
            dist_map = Utils.pred_coords_to_distmap(
                seq_encoding, res_mask, struc_out, self.struc_out_keys
            )
            recycle_data["plm_feat"] = plm_feat
            recycle_data["query_feat"] = query_feat.detach()
            recycle_data["pair_feat"] = pair_feat.detach()
            recycle_data["struc_feat"] = dist_map

        return outs, recycle_data
