"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

import copy

import numpy as np
import torch

from .base import BaseModel
from .model_util import copy_states, CropParameters
from .submodules import ResidualBlock, ConvGRU, ConvLayer
from .unet import UNetRecurrent, MultiResUNet


class E2VID(BaseModel):
    """
    E2VID architecture for image reconstruction from event-data.
    "High speed and high dynamic range video with an event camera", Rebecq et al. 2019.
    """

    def __init__(self, unet_kwargs, num_bins):
        super().__init__()
        self.crop = None

        norm = None
        use_upsample_conv = True
        final_activation = "none"
        if "norm" in unet_kwargs.keys():
            norm = unet_kwargs["norm"]
        if "use_upsample_conv" in unet_kwargs.keys():
            use_upsample_conv = unet_kwargs["use_upsample_conv"]
        if "final_activation" in unet_kwargs.keys():
            final_activation = unet_kwargs["final_activation"]

        E2VID_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": 3,
            "num_residual_blocks": 2,
            "num_output_channels": 1,
            "skip_type": "sum",
            "norm": norm,
            "num_bins": num_bins,
            "use_upsample_conv": use_upsample_conv,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "recurrent_block_type": "convlstm",
            "final_activation": final_activation,
        }

        self.num_encoders = E2VID_kwargs["num_encoders"]
        unet_kwargs.update(E2VID_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("encoding", None)  # TODO: remove
        self.unetrecurrent = UNetRecurrent(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def detach_states(self):
        detached_states = []
        for state in self.unetrecurrent.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.unetrecurrent.states = detached_states

    def reset_states(self):
        self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def forward(self, inp_voxel):
        """
        :param inp_voxel: N x num_bins x H x W
        :return: [N x 1 X H X W] reconstructed brightness signal.
        """

        # pad input
        x = inp_voxel
        if self.crop is not None:
            x = self.crop.pad(x)

        # forward pass
        img = self.unetrecurrent.forward(x)

        # crop output
        if self.crop is not None:
            img = img[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
            img = img.contiguous()

        return {"image": img}


class FireNet(BaseModel):
    """
    FireNet architecture for image reconstruction from event-data.
    "Fast image reconstruction with an event camera", Scheerlinck et al., 2019
    """

    def __init__(self, unet_kwargs, num_bins):
        super().__init__()
        base_num_channels = unet_kwargs["base_num_channels"]
        kernel_size = unet_kwargs["kernel_size"]

        padding = kernel_size // 2
        self.head = ConvLayer(num_bins, base_num_channels, kernel_size, padding=padding)
        self.G1 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R1 = ResidualBlock(base_num_channels, base_num_channels)
        self.G2 = ConvGRU(base_num_channels, base_num_channels, kernel_size)
        self.R2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred = ConvLayer(base_num_channels, out_channels=1, kernel_size=1, activation=None)
        self.num_encoders = 0  # needed by image_reconstructor.py
        self.num_recurrent_units = 2
        self.reset_states()

    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def detach_states(self):
        detached_states = []
        for state in self.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.states = detached_states

    def reset_states(self):
        self._states = [None] * self.num_recurrent_units

    def init_cropping(self, width, height):
        pass

    def forward(self, inp_voxel):
        """
        :param inp_voxel: N x num_bins x H x W
        :return: [N x 1 X H X W] reconstructed brightness signal.
        """

        # forward pass
        x = inp_voxel
        x = self.head(x)
        x = self.G1(x, self._states[0])
        self._states[0] = x
        x = self.R1(x)
        x = self.G2(x, self._states[1])
        self._states[1] = x
        x = self.R2(x)
        return {"image": self.pred(x)}


class EVFlowNet(BaseModel):
    """
    EV-FlowNet architecture for (dense/sparse) optical flow estimation from event-data.
    "EV-FlowNet: Self-Supervised Optical Flow for Event-based Cameras", Zhu et al. 2018.
    """

    def __init__(self, unet_kwargs, num_bins):
        super().__init__()
        self.crop = None
        self.mask = unet_kwargs["mask_output"]
        EVFlowNet_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": 4,
            "num_residual_blocks": 2,
            "num_output_channels": 2,
            "skip_type": "concat",
            "norm": None,
            "num_bins": num_bins,
            "use_upsample_conv": True,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "final_activation": "tanh",
        }
        self.num_encoders = EVFlowNet_kwargs["num_encoders"]
        unet_kwargs.update(EVFlowNet_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("eval", None)
        unet_kwargs.pop("encoding", None)  # TODO: remove
        unet_kwargs.pop("mask_output", None)
        unet_kwargs.pop("mask_smoothing", None)  # TODO: remove
        if "flow_scaling" in unet_kwargs.keys():
            unet_kwargs.pop("flow_scaling", None)

        self.multires_unet = MultiResUNet(unet_kwargs)

    def reset_states(self):
        pass

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def forward(self, inp_voxel, inp_cnt):
        """
        :param inp_voxel: N x num_bins x H x W
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # pad input
        x = inp_voxel
        if self.crop is not None:
            x = self.crop.pad(x)

        # forward pass
        multires_flow = self.multires_unet.forward(x)

        # upsample flow estimates to the original input resolution
        flow_list = []
        for flow in multires_flow:
            flow_list.append(
                torch.nn.functional.interpolate(
                    flow,
                    scale_factor=(
                        multires_flow[-1].shape[2] / flow.shape[2],
                        multires_flow[-1].shape[3] / flow.shape[3],
                    ),
                )
            )

        # crop output
        if self.crop is not None:
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
                flow_list[i] = flow_list[i].contiguous()

        # mask flow
        if self.mask:
            mask = torch.sum(inp_cnt, dim=1, keepdim=True)
            mask[mask > 0] = 1
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow * mask

        return {"flow": flow_list}


class FireFlowNet(BaseModel):
    """
    FireFlowNet architecture for (dense/sparse) optical flow estimation from event-data.
    "Back to Event Basics: Self Supervised Learning of Image Reconstruction from Event Data via Photometric Constancy", Paredes-Valles et al., 2020
    """

    def __init__(self, unet_kwargs, num_bins):
        super().__init__()
        base_num_channels = unet_kwargs["base_num_channels"]
        kernel_size = unet_kwargs["kernel_size"]
        self.mask = unet_kwargs["mask_output"]

        padding = kernel_size // 2
        self.E1 = ConvLayer(num_bins, base_num_channels, kernel_size, padding=padding)
        self.E2 = ConvLayer(base_num_channels, base_num_channels, kernel_size, padding=padding)
        self.R1 = ResidualBlock(base_num_channels, base_num_channels)
        self.E3 = ConvLayer(base_num_channels, base_num_channels, kernel_size, padding=padding)
        self.R2 = ResidualBlock(base_num_channels, base_num_channels)
        self.pred = ConvLayer(base_num_channels, out_channels=2, kernel_size=1, activation="tanh")

    def reset_states(self):
        pass

    def init_cropping(self, width, height):
        pass

    def forward(self, inp_voxel, inp_cnt):
        """
        :param inp_voxel: N x num_bins x H x W
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # forward pass
        x = inp_voxel
        x = self.E1(x)
        x = self.E2(x)
        x = self.R1(x)
        x = self.E3(x)
        x = self.R2(x)
        flow = self.pred(x)

        # mask flow
        if self.mask:
            mask = torch.sum(inp_cnt, dim=1, keepdim=True)
            mask[mask > 0] = 1
            flow = flow * mask

        return {"flow": [flow]}
