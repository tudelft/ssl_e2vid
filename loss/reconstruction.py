import os
import sys

import torch
import numpy as np
import torch.nn.functional as F

from .flow import AveragedIWE

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)

from utils.iwe import deblur_events
from utils.gradients import Sobel


class BrightnessConstancy(torch.nn.Module):
    """
    Self-supervised image reconstruction loss, as described in Section 3.4 of the paper 'Back to Event Basics:
    Self-Supervised Image Reconstruction for Event Cameras via Photometric Constancy', Paredes-Valles et al., CVPR'21.
    The reconstruction loss is the combination of three components.
    1) Image reconstruction through the generative model of event cameras. The reconstruction error propagates back
       through the spatial gradients of the reconstructed images. The loss consists in an L2-norm of the difference of the
       brightness increment images that can be obtained through the generative model and by means of event integration.
    2) Temporal consistency. Simple L1-norm of the warping error between two consecutive reconstructed frames.
    3) Image regularization. Conventional total variation formulation.
    """

    def __init__(self, config, device):
        super(BrightnessConstancy, self).__init__()
        self.sobel = Sobel(device)
        self.res = config["loader"]["resolution"]
        self.flow_scaling = max(config["loader"]["resolution"])
        self.weights = config["loss"]["reconstruction_regul_weight"]

        col_idx = np.linspace(0, self.res[1] - 1, num=self.res[1])
        row_idx = np.linspace(0, self.res[0] - 1, num=self.res[0])
        mx, my = np.meshgrid(col_idx, row_idx)
        indices = np.zeros((1, 2, self.res[0], self.res[1]))
        indices[:, 0, :, :] = my
        indices[:, 1, :, :] = mx
        self.indices = torch.from_numpy(indices).float().to(device)

        self.averaged_iwe = AveragedIWE(config, device)

    def generative_model(self, flow, img, inputs):
        """
        :param flow: [batch_size x 2 x H x W] optical flow map
        :param img: [batch_size x 1 x H x W] last reconstructed image
        :param inputs: dataloader dictionary
        :return generative model loss
        """

        event_cnt = inputs["inp_cnt"].to(flow.device)
        event_list = inputs["inp_list"].to(flow.device)
        pol_mask = inputs["inp_pol_mask"].to(flow.device)

        # mask optical flow with input events
        flow_mask = torch.sum(event_cnt, dim=1, keepdim=True)
        flow_mask[flow_mask > 0] = 1
        flow = flow * flow_mask

        # foward warping metrics
        warped_y = self.indices[:, 0:1, :, :] - flow[:, 1:2, :, :] * self.flow_scaling
        warped_x = self.indices[:, 1:2, :, :] - flow[:, 0:1, :, :] * self.flow_scaling
        warped_y = 2 * warped_y / (self.res[0] - 1) - 1
        warped_x = 2 * warped_x / (self.res[1] - 1) - 1
        grid_pos = torch.cat([warped_x, warped_y], dim=1).permute(0, 2, 3, 1)

        # warped predicted brightness increment (previous image)
        img_gradx, img_grady = self.sobel(img)
        warped_img_grady = F.grid_sample(img_grady, grid_pos, mode="bilinear", padding_mode="zeros")
        warped_img_gradx = F.grid_sample(img_gradx, grid_pos, mode="bilinear", padding_mode="zeros")
        pred_deltaL = warped_img_gradx * flow[:, 0:1, :, :] + warped_img_grady * flow[:, 1:2, :, :]
        pred_deltaL = pred_deltaL * self.flow_scaling

        # warped brightness increment from the averaged image of warped events
        avg_iwe = self.averaged_iwe(flow, event_list, pol_mask)
        event_deltaL = avg_iwe[:, 0:1, :, :] - avg_iwe[:, 1:2, :, :]  # C == 1

        # squared L2 norm - brightness constancy error
        bc_error = event_deltaL + pred_deltaL
        bc_error = (
            torch.norm(
                bc_error.view(
                    bc_error.shape[0],
                    bc_error.shape[1],
                    1,
                    -1,
                ),
                p=2,
                dim=3,
            )
            ** 2
        )  # norm in the spatial dimension

        return bc_error.sum()

    def temporal_consistency(self, flow, prev_img, img):
        """
        :param flow: [batch_size x 2 x H x W] optical flow map
        :param prev_img: [batch_size x 1 x H x W] previous reconstructed image
        :param img: [batch_size x 1 x H x W] last reconstructed image
        :return weighted temporal consistency loss
        """

        # foward warping metrics
        warped_y = self.indices[:, 0:1, :, :] - flow[:, 1:2, :, :] * self.flow_scaling
        warped_x = self.indices[:, 1:2, :, :] - flow[:, 0:1, :, :] * self.flow_scaling
        warped_y = 2 * warped_y / (self.res[0] - 1) - 1
        warped_x = 2 * warped_x / (self.res[1] - 1) - 1
        grid_pos = torch.cat([warped_x, warped_y], dim=1).permute(0, 2, 3, 1)

        # temporal consistency
        warped_prev_img = F.grid_sample(prev_img, grid_pos, mode="bilinear", padding_mode="zeros")
        tc_error = img - warped_prev_img
        tc_error = (
            torch.norm(
                tc_error.view(
                    tc_error.shape[0],
                    tc_error.shape[1],
                    1,
                    -1,
                ),
                p=1,
                dim=3,
            )
            ** 1
        )  # norm in the spatial dimension
        tc_error = tc_error.sum()

        return self.weights[1] * tc_error

    def regularization(self, img):
        """
        :param img: [batch_size x 1 x H x W] last reconstructed image
        :return weighted image regularization loss
        """

        # conventional total variation with forward differences
        img_dx = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        img_dy = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        tv_error = img_dx.sum() + img_dy.sum()

        return self.weights[0] * tv_error
