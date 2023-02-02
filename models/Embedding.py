#!/usr/bin/env python3
# encoding: utf-8

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def onehot(idx: torch.Tensor, n_alphabet):
    return torch.eye(n_alphabet + 1)[idx, :][..., :n_alphabet].to(idx.device)


class RelativePositionalEncoding2D(nn.Module):
    """
    Relative Positional Encoding.
    """

    def __init__(self, n_emb: int, max_gap: int = 32, *args, **kwargs):
        super().__init__()
        self.max_gap = max_gap
        self.n_index = max_gap * 2 + 1
        self.pos_emb = nn.Linear(self.n_index, n_emb)

    def forward(self, idxs):
        """
        idxs: (n_batch, seq_len)
        return: (n_batch, seq_len, seq_len, n_emb)
        """
        pos_2d = idxs[:, None, :] - idxs[:, :, None]
        pos_2d = pos_2d.clip(min=-self.max_gap, max=self.max_gap) + self.max_gap

        pos_2d_onehot = torch.eye(self.n_index)[pos_2d.long(), :].type_as(
            self.pos_emb.weight
        )

        return self.pos_emb(pos_2d_onehot)


class EmbeddingModule(nn.Module):
    """
    Generate the input embedding.
    """

    def __init__(
        self,
        n_alphabet: int = 23,
        n_emb_seq: int = 256,
        n_emb_pair: int = 128,
        max_gap: int = 32,
        plm_config: dict = {},
        struc_dist_bins: list = [],
        *args,
        **kwargs,
    ):
        super().__init__()

        self.n_alphabet = n_alphabet

        self.struc_dist_bins = torch.tensor(struc_dist_bins)
        self.n_struc_dist_bins = len(self.struc_dist_bins)

        # query emb
        self.query_emb_layer = nn.Linear(n_alphabet, n_emb_seq)

        # pair emb
        self.left_linear = nn.Linear(n_alphabet, n_emb_pair)
        self.right_linear = nn.Linear(n_alphabet, n_emb_pair)

        # pair positional emb
        self.PosEmb_pair_layer = RelativePositionalEncoding2D(
            n_emb_pair, max_gap=max_gap
        )

        self.query_recycle_norm = nn.LayerNorm(n_emb_seq)
        self.pair_recycle_norm = nn.LayerNorm(n_emb_pair)
        self.struc_emb_layer = nn.Linear(self.n_struc_dist_bins + 1, n_emb_pair)

        # protein language model
        self.pretraining_layer = PLMModule(
            n_emb_seq,
            n_emb_pair,
            plm_models=plm_config["plm_models"],
            model_config=plm_config["model_config"],
        )

    def forward(
        self,
        seq_encoding,
        ESM_token=None,
        ProtTrans_token=None,
        recycle_data=None,
        res_mask=None,
        *args,
        **kwargs,
    ):

        n_batch, seq_len = seq_encoding.shape[:2]

        seq_onehot = onehot(seq_encoding, self.n_alphabet)
        seq_emb = self.query_emb_layer(seq_onehot)
        seq_emb = seq_emb.unsqueeze(1)

        # pair emb
        x_left = self.left_linear(seq_onehot)
        x_right = self.right_linear(seq_onehot)
        pair_emb = x_left[:, :, None] + x_right[:, None]

        # PosEmb_pair_layer
        res_idxs = torch.tensor([i for i in range(seq_len)]).unsqueeze(0)
        pair_emb = pair_emb + self.PosEmb_pair_layer(res_idxs).type_as(pair_emb)

        # recycle feature
        # query_feat
        prev_query_feat = (
            recycle_data["query_feat"]
            if recycle_data is not None
            else torch.zeros_like(seq_emb[:, 0, :, :])
        )
        prev_query_feat = self.query_recycle_norm(prev_query_feat.nan_to_num())
        seq_emb[:, 0, :, :] = seq_emb[:, 0, :, :] + prev_query_feat

        # struc_feat
        if recycle_data is not None:
            dist_map = recycle_data["struc_feat"]
            dist_idx = torch.bucketize(
                dist_map, self.struc_dist_bins.type_as(pair_emb), right=True
            )
        else:
            dist_idx = torch.zeros((n_batch, seq_len, seq_len), dtype=torch.int64)
            dist_idx = dist_idx.to(pair_emb.device)

        dist_onehot = onehot(dist_idx, self.n_struc_dist_bins + 1)
        prev_struc_feat = self.struc_emb_layer(dist_onehot)

        # pair_feat
        prev_pair_feat = (
            recycle_data["pair_feat"]
            if recycle_data is not None
            else torch.zeros_like(pair_emb)
        )
        prev_pair_feat = self.pair_recycle_norm(prev_pair_feat.nan_to_num())
        pair_emb = pair_emb + prev_pair_feat + prev_struc_feat

        plm_feat = recycle_data["plm_feat"] if recycle_data is not None else {}
        seq_emb, pair_emb, plm_feat = self.pretraining_layer(
            seq_emb, pair_emb, ESM_token, ProtTrans_token, plm_feat, res_mask
        )

        return seq_emb, pair_emb, plm_feat


class PLMModule(nn.Module):
    def __init__(
        self,
        n_emb_query: int = 256,
        n_emb_pair: int = 128,
        plm_models: list = [],
        model_config: dict = {},
    ):
        super().__init__()

        self.plm_models = plm_models
        self.model_config = model_config
        self.only_esm1b = len(self.plm_models) == 1 and "ESM1b" in self.plm_models
        device = model_config["device"]

        # load param
        if "ESM1b" in plm_models:
            import esm

            self.ESM1b_model = esm.pretrained.load_model_and_alphabet_core(
                torch.load(os.environ["ESM1b_param"], map_location=device), None
            )[0]

            self.ESM1b_linear = nn.Linear(
                model_config["ESM1b"]["n_embedding"], n_emb_query
            )
            self.ESM1b_attn_linear = nn.Linear(
                model_config["ESM1b"]["n_attn_embedding"], n_emb_pair
            )

        if "ESM1v" in plm_models:
            import esm

            self.ESM1v_model = esm.pretrained.load_model_and_alphabet_core(
                torch.load(os.environ["ESM1v_param"], map_location=device), None
            )[0]

            self.ESM1v_linear = nn.Linear(
                model_config["ESM1v"]["n_embedding"], n_emb_query
            )
            self.ESM1v_attn_linear = nn.Linear(
                model_config["ESM1v"]["n_attn_embedding"], n_emb_pair
            )

        if "ProtTrans" in plm_models:
            from transformers import T5EncoderModel

            self.ProtTrans_model = T5EncoderModel.from_pretrained(
                os.environ["ProtTrans_param"]
            )
            self.ProtTrans_model = self.ProtTrans_model.to(device)

            self.ProtTrans_linear = nn.Linear(
                model_config["ProtTrans"]["n_embedding"], n_emb_query
            )

            self.ProtTrans_attn_linear = nn.Linear(
                model_config["ProtTrans"]["n_attn_embedding"], n_emb_pair
            )

        if not self.only_esm1b:
            self.norm_pair_out = nn.LayerNorm(n_emb_pair)

    def get_ESM1b_emb(self, toks):
        """
        toks: [batch, seq_len+2]
        """
        # [batch, seq_len, n_emb]
        results = self.ESM1b_model(
            toks.long(), repr_layers=[33], need_head_weights=True
        )

        ESM_emb = results["representations"][33][:, 1:-1, :].nan_to_num()

        attn_layers = self.model_config["ESM1b"]["attn_layers"]
        ESM_attn = (
            torch.cat(
                [results["attentions"][:, i, :, 1:-1, 1:-1] for i in attn_layers], dim=1
            )
            .permute(0, 2, 3, 1)
            .nan_to_num()
        )

        return ESM_emb, ESM_attn

    def get_ESM1v_emb(self, toks):
        """
        toks: [batch, seq_len+2]
        """
        results = self.ESM1v_model(
            toks.long(), repr_layers=[33], need_head_weights=True
        )

        ESM_emb = results["representations"][33][:, 1:-1, :].nan_to_num()

        attn_layers = self.model_config["ESM1v"]["attn_layers"]
        ESM_attn = (
            torch.cat(
                [results["attentions"][:, i, :, 1:-1, 1:-1] for i in attn_layers], dim=1
            )
            .permute(0, 2, 3, 1)
            .nan_to_num()
        )

        return ESM_emb, ESM_attn

    def get_ProtTrans_emb(self, ProtTrans_token, res_mask):
        results = self.ProtTrans_model(
            input_ids=ProtTrans_token.long(),
            attention_mask=F.pad(res_mask, (1, 0), "constant", 1),
            output_attentions=True,
        )

        ProtTrans_emb = results["last_hidden_state"][:, :-1, :].nan_to_num()

        attn_layers = self.model_config["ProtTrans"]["attn_layers"]
        ProtTrans_attn = torch.cat(
            [results["attentions"][i][:, :, :-1, :-1] for i in attn_layers], dim=1
        )
        ProtTrans_attn = ProtTrans_attn.permute(0, 2, 3, 1).nan_to_num()

        return ProtTrans_emb, ProtTrans_attn

    def forward(
        self,
        seq_emb,
        pair_emb,
        ESM_token,
        ProtTrans_token,
        recycle_data,
        res_mask,
        *args,
        **kwargs,
    ):
        query_emb = seq_emb[:, 0, :, :]

        # seq_emb
        if "ESM1b" in self.plm_models:
            if "ESM1b_emb" in recycle_data:
                ESM1b_emb = recycle_data["ESM1b_emb"]
                ESM1b_attn = recycle_data["ESM1b_attn"]
            else:
                ESM1b_emb, ESM1b_attn = self.get_ESM1b_emb(ESM_token)
                recycle_data["ESM1b_emb"] = ESM1b_emb
                recycle_data["ESM1b_attn"] = ESM1b_attn

            query_emb += self.ESM1b_linear(ESM1b_emb)
            pair_emb += self.ESM1b_attn_linear(ESM1b_attn)

        if "ESM1v" in self.plm_models:
            if "ESM1v_emb" in recycle_data:
                ESM1v_emb = recycle_data["ESM1v_emb"]
                ESM1v_attn = recycle_data["ESM1v_attn"]
            else:
                ESM1v_emb, ESM1v_attn = self.get_ESM1v_emb(ESM_token)
                recycle_data["ESM1v_emb"] = ESM1v_emb
                recycle_data["ESM1v_attn"] = ESM1v_attn

            query_emb += self.ESM1v_linear(ESM1v_emb)
            pair_emb += self.ESM1v_attn_linear(ESM1v_attn)

        if "ProtTrans" in self.plm_models:
            if "ProtTrans_emb" in recycle_data:
                ProtTrans_emb = recycle_data["ProtTrans_emb"]
                ProtTrans_attn = recycle_data["ProtTrans_attn"]
            else:
                ProtTrans_emb, ProtTrans_attn = self.get_ProtTrans_emb(
                    ProtTrans_token, res_mask
                )
                recycle_data["ProtTrans_emb"] = ProtTrans_emb
                recycle_data["ProtTrans_attn"] = ProtTrans_attn

            query_emb += self.ProtTrans_linear(ProtTrans_emb)
            pair_emb += self.ProtTrans_attn_linear(ProtTrans_attn)

        if not self.only_esm1b:
            pair_emb = self.norm_pair_out(pair_emb)

        seq_emb[:, 0, :, :] = query_emb

        return seq_emb, pair_emb, recycle_data
