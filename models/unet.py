"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

import torch
import torch.nn as nn

from .model_util import *
from .submodules import (
    ConvGRU,
    ConvLayer,
    RecurrentConvLayer,
    ResidualBlock,
    TransposedConvLayer,
    UpsampleConvLayer,
)


class BaseUNet(nn.Module):
    """
    Base class for conventional UNet architecture.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(
        self,
        base_num_channels,
        num_encoders,
        num_residual_blocks,
        num_output_channels,
        skip_type,
        norm,
        use_upsample_conv,
        num_bins,
        recurrent_block_type=None,
        kernel_size=5,
        channel_multiplier=2,
    ):
        super(BaseUNet, self).__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = norm
        self.num_bins = num_bins
        self.recurrent_block_type = recurrent_block_type
        self.channel_multiplier = channel_multiplier

        self.skip_ftn = eval("skip_" + skip_type)
        if use_upsample_conv:
            self.UpsampleLayer = UpsampleConvLayer
        else:
            self.UpsampleLayer = TransposedConvLayer
        assert self.num_output_channels > 0

        self.encoder_input_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i)) for i in range(self.num_encoders)
        ]
        self.encoder_output_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i + 1)) for i in range(self.num_encoders)
        ]
        self.max_num_channels = self.encoder_output_sizes[-1]

    def build_encoders(self):
        encoders = nn.ModuleList()
        for (input_size, output_size) in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            encoders.append(
                ConvLayer(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.kernel_size // 2,
                    activation=self.activation,
                    norm=self.norm,
                )
            )
        return encoders

    def build_resblocks(self):
        resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))
        return resblocks

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(
                self.UpsampleLayer(
                    input_size if self.skip_type == "sum" else 2 * input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    norm=self.norm,
                )
            )
        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return ConvLayer(
            self.base_num_channels if self.skip_type == "sum" else 2 * self.base_num_channels,
            num_output_channels,
            1,
            activation=None,
            norm=norm,
        )


class UNetRecurrent(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, unet_kwargs):
        final_activation = unet_kwargs.pop("final_activation", "none")
        self.final_activation = getattr(torch, final_activation, None)
        unet_kwargs["num_output_channels"] = 1
        super().__init__(**unet_kwargs)

        self.head = ConvLayer(
            self.num_bins,
            self.base_num_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )

        self.encoders = self.build_recurrent_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_decoders()
        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)
        self.states = [None] * self.num_encoders

    def build_recurrent_encoders(self):
        encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            encoders.append(
                RecurrentConvLayer(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.kernel_size // 2,
                    recurrent_block_type=self.recurrent_block_type,
                    norm=self.norm,
                )
            )
        return encoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i])
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.pred(self.skip_ftn(x, head))
        if self.final_activation is not None:
            img = self.final_activation(img)
        return img


class MultiResUNet(BaseUNet):
    """
    Conventional UNet architecture.
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    """

    def __init__(self, unet_kwargs):
        self.final_activation = unet_kwargs.pop("final_activation", "none")
        self.skip_type = "concat"
        super().__init__(**unet_kwargs)

        self.encoders = self.build_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_multires_prediction_decoders()
        self.preds = self.build_multires_prediction_layer()

    def build_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            encoders.append(
                ConvLayer(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.kernel_size // 2,
                    norm=self.norm,
                )
            )
        return encoders

    def build_multires_prediction_layer(self):
        preds = nn.ModuleList()
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        for output_size in decoder_output_sizes:
            preds.append(
                ConvLayer(output_size, self.num_output_channels, 1, activation=self.final_activation, norm=self.norm)
            )
        return preds

    def build_multires_prediction_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            prediction_channels = 0 if i == 0 else self.num_output_channels
            decoders.append(
                self.UpsampleLayer(
                    2 * input_size + prediction_channels,
                    output_size,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    norm=self.norm,
                )
            )
        return decoders

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_ftn(x, blocks[self.num_encoders - i - 1])
            if i > 0:
                x = self.skip_ftn(predictions[-1], x)
            x = decoder(x)
            predictions.append(pred(x))

        return predictions
