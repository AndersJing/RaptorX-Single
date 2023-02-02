#!/usr/bin/env python3
# encoding: utf-8

import torch.nn as nn

from .Layer import activation_func, normalization_func


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size and dilation
        self.padding = (
            self.dilation[0] * (self.kernel_size[0] - 1) // 2,
            self.dilation[1] * (self.kernel_size[1] - 1) // 2,
        )


class ResNet2DBlock(nn.Module):
    """
    ResNet 2D Residual Block
    """

    def __init__(
        self,
        n_input,
        n_output,
        kernel_size=3,
        dilation=1,
        dropout=0.0,
        activation="elu",
        normalization="instance",
        bias=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        # if use checkpoint, must use Linear see https://github.com/pytorch/pytorch/issues/48439
        self.skip_connection = (
            nn.Sequential(
                Conv2dAuto(n_input, n_output, kernel_size=1, bias=bias),
                normalization_func("2D", normalization, n_output),
                # activation_func(activation),
            )
            if n_input != n_output
            else None
        )

        self.block = nn.Sequential(
            Conv2dAuto(
                n_input, n_output, kernel_size=kernel_size, dilation=dilation, bias=bias
            ),
            normalization_func("2D", normalization, n_output),
            activation_func(activation),
            nn.Dropout2d(p=dropout),
            Conv2dAuto(
                n_output,
                n_output,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=bias,
            ),
            normalization_func("2D", normalization, n_output),
        )

        self.activate = activation_func(activation)

    def forward(self, x):
        residual = x
        x = self.block(x)
        if self.skip_connection is not None:
            residual = self.skip_connection(residual)
        x += residual
        x = self.activate(x)
        return x


class ResNet2D(nn.Module):
    """
    ResNet2D layer composed by 'n' blocks stacked one after the other
    """

    def __init__(
        self,
        n_input,
        n_channel,
        n_block,
        n_output=None,
        kernel_size=3,
        dilation=[1],
        dropout=0.0,
        activation="elu",
        normalization="instance",
        bias=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        # assert n_block%len(dilation) == 0, 'the dilation num must be devided by n_block.'
        self.n_channel = n_channel
        self.n_output = n_channel if n_output is None else n_output

        self.proj_in = (
            nn.Sequential(
                Conv2dAuto(n_input, n_channel, kernel_size=1, dilation=1, bias=bias),
                normalization_func("2D", normalization, n_channel),
                activation_func(activation),
            )
            if n_input != n_channel
            else None
        )

        self.blocks = nn.ModuleList()
        dilations = [dilation[_ % (len(dilation))] for _ in range(n_block)]
        self.blocks.extend(
            [
                *[
                    ResNet2DBlock(
                        n_channel,
                        n_channel,
                        kernel_size=kernel_size,
                        dilation=_dilation,
                        dropout=dropout,
                        activation=activation,
                        normalization=normalization,
                        bias=bias,
                        *args,
                        **kwargs,
                    )
                    for _dilation in dilations
                ]
            ]
        )

        self.proj_out = (
            nn.Conv2d(n_channel, n_output, 1) if self.n_output != n_channel else None
        )

    def get_output(self):
        return self.n_output

    def forward(self, x, *args, **kwargs):
        """
        Input  (B, C, H, W)
        Output (B, C, H, W)
        """

        if self.proj_in is not None:
            x = self.proj_in(x)

        for i, block in enumerate(self.blocks):
            x = block(x, **kwargs)

        if self.proj_out is not None:
            x = self.proj_out(x)

        return x
