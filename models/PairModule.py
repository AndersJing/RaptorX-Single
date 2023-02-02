#!/usr/bin/env python3
# encoding: utf-8

import torch
import torch.nn as nn

from .ResNet2D import ResNet2D


class OutputLayer(nn.Module):
    def __init__(self, n_input, pred_config, *args, **kwargs):
        super().__init__()

        self.symmetric_out = []
        self.output_layers = nn.ModuleDict({})
        for pred_name in pred_config:
            _n_output = pred_config[pred_name]["n_output"]
            self.output_layers.update(
                {pred_name: ResNet2D(n_input, _n_output * 2, 1, _n_output)}
            )
            if pred_config[pred_name]["symmetric_out"]:
                self.symmetric_out.append(pred_name)

    def forward(self, x):
        # x: (B, C, H, W)
        outputs = ()
        for pred_name in self.output_layers:
            out = self.output_layers[pred_name](x)
            if pred_name in self.symmetric_out:
                out = (out + out.transpose(2, 3)) * 0.5
            outputs += (out,)

        return outputs


class PairModule(nn.Module):
    def __init__(self, n_input, configs):
        super().__init__()

        self.pair_norm = nn.LayerNorm(n_input)
        self.output_layer = OutputLayer(n_input, configs["pred_config"])

    def forward(self, x):
        # (B, H, W, C)
        x = self.pair_norm(x)

        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        outputs = self.output_layer(x)

        # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()

        return outputs, x
