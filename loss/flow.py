import os
import sys

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)

from utils.iwe import purge_unfeasible, get_interpolation, interpolate


class EventWarping(nn.Module):
    """
    Contrast maximization loss, as described in Section 3.2 of the paper 'Unsupervised Event-based Learning
    of Optical Flow, Depth, and Egomotion', Zhu et al., CVPR'19.
    The contrast maximization loss is the minimization of the per-pixel and per-polarity image of averaged
    timestamps of the input events after they have been compensated for their motion using the estimated
    optical flow. This minimization is performed in a forward and in a backward fashion to prevent scaling
    issues during backpropagation.
    """

    def __init__(self, config, device):
        super(EventWarping, self).__init__()
        self.res = config["loader"]["resolution"]
        self.flow_scaling = max(config["loader"]["resolution"])
        self.weight = config["loss"]["flow_regul_weight"]
        self.device = device

    def forward(self, flow_list, event_list, pol_mask):
        """
        :param flow_list: [[batch_size x 2 x H x W]] list of optical flow maps
        :param event_list: [batch_size x N x 4] input events (y, x, ts, p)
        :param pol_mask: [batch_size x N x 2] per-polarity binary mask of the input events
        """

        # split input
        pol_mask = torch.cat([pol_mask for i in range(4)], dim=1)
        ts_list = torch.cat([event_list[:, :, 0:1] for i in range(4)], dim=1)

        # flow vector per input event
        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        loss = 0
        for flow in flow_list:

            # get flow for every event in the list
            flow = flow.view(flow.shape[0], 2, -1)
            event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
            event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
            event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
            event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
            event_flow = torch.cat([event_flowy, event_flowx], dim=2)

            # interpolate forward
            tref = 1
            fw_idx, fw_weights = get_interpolation(event_list, event_flow, tref, self.res, self.flow_scaling)

            # per-polarity image of (forward) warped events
            fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 0:1])
            fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 1:2])

            # image of (forward) warped averaged timestamps
            fw_iwe_pos_ts = interpolate(
                fw_idx.long(), fw_weights * ts_list, self.res, polarity_mask=pol_mask[:, :, 0:1]
            )
            fw_iwe_neg_ts = interpolate(
                fw_idx.long(), fw_weights * ts_list, self.res, polarity_mask=pol_mask[:, :, 1:2]
            )
            fw_iwe_pos_ts /= fw_iwe_pos + 1e-9
            fw_iwe_neg_ts /= fw_iwe_neg + 1e-9

            # interpolate backward
            tref = 0
            bw_idx, bw_weights = get_interpolation(event_list, event_flow, tref, self.res, self.flow_scaling)

            # per-polarity image of (backward) warped events
            bw_iwe_pos = interpolate(bw_idx.long(), bw_weights, self.res, polarity_mask=pol_mask[:, :, 0:1])
            bw_iwe_neg = interpolate(bw_idx.long(), bw_weights, self.res, polarity_mask=pol_mask[:, :, 1:2])

            # image of (backward) warped averaged timestamps
            bw_iwe_pos_ts = interpolate(
                bw_idx.long(), bw_weights * (1 - ts_list), self.res, polarity_mask=pol_mask[:, :, 0:1]
            )
            bw_iwe_neg_ts = interpolate(
                bw_idx.long(), bw_weights * (1 - ts_list), self.res, polarity_mask=pol_mask[:, :, 1:2]
            )
            bw_iwe_pos_ts /= bw_iwe_pos + 1e-9
            bw_iwe_neg_ts /= bw_iwe_neg + 1e-9

            # flow smoothing
            flow = flow.view(flow.shape[0], 2, self.res[0], self.res[1])
            flow_dx = flow[:, :, :-1, :] - flow[:, :, 1:, :]
            flow_dy = flow[:, :, :, :-1] - flow[:, :, :, 1:]
            flow_dx = torch.sqrt(flow_dx ** 2 + 1e-6)  # charbonnier
            flow_dy = torch.sqrt(flow_dy ** 2 + 1e-6)  # charbonnier

            loss += (
                torch.sum(fw_iwe_pos_ts ** 2)
                + torch.sum(fw_iwe_neg_ts ** 2)
                + torch.sum(bw_iwe_pos_ts ** 2)
                + torch.sum(bw_iwe_neg_ts ** 2)
                + self.weight * (flow_dx.sum() + flow_dy.sum())
            )

        return loss


class AveragedIWE(nn.Module):
    """
    Returns an image of the per-pixel and per-polarity average number of warped events given
    an optical flow map.
    """

    def __init__(self, config, device):
        super(AveragedIWE, self).__init__()
        self.res = config["loader"]["resolution"]
        self.flow_scaling = max(config["loader"]["resolution"])
        self.batch_size = config["loader"]["batch_size"]
        self.device = device

    def forward(self, flow, event_list, pol_mask):
        """
        :param flow: [batch_size x 2 x H x W] optical flow maps
        :param event_list: [batch_size x N x 4] input events (y, x, ts, p)
        :param pol_mask: [batch_size x N x 2] per-polarity binary mask of the input events
        """

        # original location of events
        idx = event_list[:, :, 1:3].clone()
        idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        idx = torch.sum(idx, dim=2, keepdim=True)

        # flow vector per input event
        flow_idx = event_list[:, :, 1:3].clone()
        flow_idx[:, :, 0] *= self.res[1]  # torch.view is row-major
        flow_idx = torch.sum(flow_idx, dim=2)

        # get flow for every event in the list
        flow = flow.view(flow.shape[0], 2, -1)
        event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
        event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
        event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
        event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
        event_flow = torch.cat([event_flowy, event_flowx], dim=2)

        # interpolate forward
        fw_idx, fw_weights = get_interpolation(event_list, event_flow, 1, self.res, self.flow_scaling, round_idx=True)

        # per-polarity image of (forward) warped events
        fw_iwe_pos = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 0:1])
        fw_iwe_neg = interpolate(fw_idx.long(), fw_weights, self.res, polarity_mask=pol_mask[:, :, 1:2])
        if fw_idx.shape[1] == 0:
            return torch.cat([fw_iwe_pos, fw_iwe_neg], dim=1)

        # make sure unfeasible mappings are not considered
        pol_list = event_list[:, :, 3:4].clone()
        pol_list[pol_list < 1] = 0  # negative polarity set to 0
        pol_list[fw_weights == 0] = 2  # fake polarity to detect unfeasible mappings

        # encode unique ID for pixel location mapping (idx <-> fw_idx = m_idx)
        m_idx = torch.cat([idx.long(), fw_idx.long()], dim=2)
        m_idx[:, :, 0] *= self.res[0] * self.res[1]
        m_idx = torch.sum(m_idx, dim=2, keepdim=True)

        # encode unique ID for per-polarity pixel location mapping (pol_list <-> m_idx = pm_idx)
        pm_idx = torch.cat([pol_list.long(), m_idx.long()], dim=2)
        pm_idx[:, :, 0] *= (self.res[0] * self.res[1]) ** 2
        pm_idx = torch.sum(pm_idx, dim=2, keepdim=True)

        # number of different pixels locations from where pixels originate during warping
        # this needs to be done per batch as the number of unique indices differs
        fw_iwe_pos_contrib = torch.zeros((flow.shape[0], self.res[0] * self.res[1], 1)).to(self.device)
        fw_iwe_neg_contrib = torch.zeros((flow.shape[0], self.res[0] * self.res[1], 1)).to(self.device)
        for b in range(0, self.batch_size):

            # per-polarity unique mapping combinations
            unique_pm_idx = torch.unique(pm_idx[b, :, :], dim=0)
            unique_pm_idx = torch.cat(
                [
                    unique_pm_idx // ((self.res[0] * self.res[1]) ** 2),
                    unique_pm_idx % ((self.res[0] * self.res[1]) ** 2),
                ],
                dim=1,
            )  # (pol_idx, mapping_idx)
            unique_pm_idx = torch.cat(
                [unique_pm_idx[:, 0:1], unique_pm_idx[:, 1:2] % (self.res[0] * self.res[1])], dim=1
            )  # (pol_idx, fw_idx)
            unique_pm_idx[:, 0] *= self.res[0] * self.res[1]
            unique_pm_idx = torch.sum(unique_pm_idx, dim=1, keepdim=True)

            # per-polarity unique receiving pixels
            unique_pfw_idx, contrib_pfw = torch.unique(unique_pm_idx[:, 0], dim=0, return_counts=True)
            unique_pfw_idx = unique_pfw_idx.view((unique_pfw_idx.shape[0], 1))
            contrib_pfw = contrib_pfw.view((contrib_pfw.shape[0], 1))
            unique_pfw_idx = torch.cat(
                [unique_pfw_idx // (self.res[0] * self.res[1]), unique_pfw_idx % (self.res[0] * self.res[1])],
                dim=1,
            )  # (polarity mask, fw_idx)

            # positive scatter pixel contribution
            mask_pos = unique_pfw_idx[:, 0:1].clone()
            mask_pos[mask_pos == 2] = 0  # remove unfeasible mappings
            b_fw_iwe_pos_contrib = torch.zeros((self.res[0] * self.res[1], 1)).to(self.device)
            b_fw_iwe_pos_contrib = b_fw_iwe_pos_contrib.scatter_add_(
                0, unique_pfw_idx[:, 1:2], mask_pos.float() * contrib_pfw.float()
            )

            # negative scatter pixel contribution
            mask_neg = unique_pfw_idx[:, 0:1].clone()
            mask_neg[mask_neg == 2] = 1  # remove unfeasible mappings
            mask_neg = 1 - mask_neg  # invert polarities
            b_fw_iwe_neg_contrib = torch.zeros((self.res[0] * self.res[1], 1)).to(self.device)
            b_fw_iwe_neg_contrib = b_fw_iwe_neg_contrib.scatter_add_(
                0, unique_pfw_idx[:, 1:2], mask_neg.float() * contrib_pfw.float()
            )

            # store info
            fw_iwe_pos_contrib[b, :, :] = b_fw_iwe_pos_contrib
            fw_iwe_neg_contrib[b, :, :] = b_fw_iwe_neg_contrib

        # average number of warped events per pixel
        fw_iwe_pos_contrib = fw_iwe_pos_contrib.view((flow.shape[0], 1, self.res[0], self.res[1]))
        fw_iwe_neg_contrib = fw_iwe_neg_contrib.view((flow.shape[0], 1, self.res[0], self.res[1]))
        fw_iwe_pos[fw_iwe_pos_contrib > 0] /= fw_iwe_pos_contrib[fw_iwe_pos_contrib > 0]
        fw_iwe_neg[fw_iwe_neg_contrib > 0] /= fw_iwe_neg_contrib[fw_iwe_neg_contrib > 0]

        return torch.cat([fw_iwe_pos, fw_iwe_neg], dim=1)
