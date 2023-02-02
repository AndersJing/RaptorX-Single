#!/usr/bin/env python3
# encoding: utf-8

import numpy as np


CONFIGS = {
    "n_alphabet": 20 + 1 + 1 + 1,  # AA + X + gap + mask
    "EMBEDDING_MODULE": {
        "plm_config": {
            "model_config": {
                "ESM1b": {
                    "n_embedding": 1280,
                    "attn_layers": [31, 32],
                    "n_attn_embedding": 40,
                },
                "ESM1v": {
                    "n_embedding": 1280,
                    "attn_layers": [31, 32],
                    "n_attn_embedding": 40,
                },
                "ProtTrans": {
                    "n_embedding": 1024,
                    "attn_layers": [22, 23],
                    "n_attn_embedding": 32 * 2,
                },
                "device": "cpu",
            },  # config of self-train model
        },
        "struc_dist_bins": np.linspace(3, 20.0, num=18).tolist(),
    },
    "SEQ_MODULE": {
        "seq_config": {
            "activation": "relu",
            "head_by_head": True,
            "seqlen_scaling": False,
            "bias": False,
            "n_head": 8,
            "n_ff_hidden": 1024,
            "attn_config": {
                "attn_bias": True,
                "pair_feat_norm": True,
                "gate_on_V": True,
                "pair_feat_dim": 128,
            },
        },
        "pair_config": {
            "emb_dim": 128,
            "TriangleAttn_config": {
                "attn_config": {
                    "attn_bias": True,
                    "pair_feat_norm": False,
                    "gate_on_V": True,
                    "pair_feat_dim": 128,
                },
                "n_head": 4,
            },
            "activation": "relu",
        },
    },
    "PAIR_MODULE": {
        "pred_config": {
            "CbCb": {"symmetric_out": True, "n_output": 37},
            "Ca1Cb1Cb2Ca2": {"symmetric_out": True, "n_output": 25},
            "N1Ca1Cb1Cb2": {"symmetric_out": False, "n_output": 25},
            "Ca1Cb1Cb2": {"symmetric_out": False, "n_output": 13},
        },
    },
    "STRUC_MODULE": {
        "out_keys": [
            "Ca_coord",
            "orient",
            "aux_Ca_coord",
            "aux_orient",
            "unnorm_angles",
            "norm_angles",
            "aux_unnorm_angles",
            "aux_norm_angles",
            "pLDDT_logit",
        ],
    },
    "PREDICTION": {
        "pred_keys": [
            "CbCb",
            "Ca1Cb1Cb2Ca2",
            "N1Ca1Cb1Cb2",
            "Ca1Cb1Cb2",
            "Ca_coord",
            "orient",
            "aux_Ca_coord",
            "aux_orient",
            "unnorm_angles",
            "norm_angles",
            "aux_unnorm_angles",
            "aux_norm_angles",
            "pLDDT_logit",
        ]
    },
}
