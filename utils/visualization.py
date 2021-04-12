import os
import cv2
import matplotlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Visualization:
    """
    Utility class for the visualization and storage of rendered image-like representation
    of multiple elements of the optical flow estimation and image reconstruction pipeline.
    """

    def __init__(self, kwargs, eval_id=-1):
        self.img_idx = 0
        self.px = kwargs["vis"]["px"]
        self.color_scheme = "green_red"  # gray / blue_red / green_red

        if eval_id >= 0:
            self.store_dir = kwargs["trained_model"] + "results/"
            self.store_dir = self.store_dir + "eval_" + str(eval_id) + "/"
            if not os.path.exists(self.store_dir):
                os.makedirs(self.store_dir)
            self.store_file = None

    def update(self, inputs, flow, iwe, brightness):
        """
        Live visualization.
        :param inputs: dataloader dictionary
        :param flow: [batch_size x 2 x H x W] optical flow map
        :param iwe: [batch_size x 1 x H x W] image of warped events
        :param brightness: [batch_size x 1 x H x W] reconstructed image
        """

        inp_events = inputs["inp_cnt"] if "inp_cnt" in inputs.keys() else None
        inp_frames = inputs["inp_frames"] if "inp_frames" in inputs.keys() else None
        height = inp_events.shape[2]
        width = inp_events.shape[3]

        # input events
        inp_events = inp_events.detach()
        inp_events_npy = inp_events.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
        cv2.namedWindow("Input Events", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Input Events", int(self.px), int(self.px))
        cv2.imshow("Input Events", self.events_to_image(inp_events_npy))

        # input frames
        if inp_frames is not None:
            frame_image = np.zeros((height, 2 * width))
            inp_frames = inp_frames.detach()
            inp_frames_npy = inp_frames.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            frame_image[:height, 0:width] = inp_frames_npy[:, :, 0] / 255.0
            frame_image[:height, width : 2 * width] = inp_frames_npy[:, :, 1] / 255.0
            cv2.namedWindow("Input Frames (Prev/Curr)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Input Frames (Prev/Curr)", int(2 * self.px), int(self.px))
            cv2.imshow("Input Frames (Prev/Curr)", frame_image)

        # optical flow
        if flow is not None:
            flow = flow.detach()
            flow_npy = flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            flow_npy = self.flow_to_image(flow_npy[:, :, 0], flow_npy[:, :, 1])
            flow_npy = cv2.cvtColor(flow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Estimated Flow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Estimated Flow", int(self.px), int(self.px))
            cv2.imshow("Estimated Flow", flow_npy)

        # image of warped events
        if iwe is not None:
            iwe = iwe.detach()
            iwe_npy = iwe.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            iwe_npy = self.events_to_image(iwe_npy)
            cv2.namedWindow("Image of Warped Events", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image of Warped Events", int(self.px), int(self.px))
            cv2.imshow("Image of Warped Events", iwe_npy)

        # reconstructed brightness
        if brightness is not None:
            brightness = brightness.detach()
            brightness_npy = brightness.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 1))
            intensity_npy = brightness_npy.reshape((height, width, 1))
            intensity_image = self.minmax_norm(intensity_npy)
            cv2.namedWindow("Reconstructed Intensity", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Reconstructed Intensity", int(self.px), int(self.px))
            cv2.imshow("Reconstructed Intensity", intensity_image)

        cv2.waitKey(1)

    def store(self, inputs, flow, iwe, brightness, sequence, ts=None):
        """
        Store rendered images.
        :param inputs: dataloader dictionary
        :param flow: [batch_size x 2 x H x W] optical flow map
        :param iwe: [batch_size x 1 x H x W] image of warped events
        :param brightness: [batch_size x 1 x H x W] reconstructed image
        :param sequence: filename of the event sequence under analysis
        :param ts: timestamp associated with rendered files (default = None)
        """

        inp_events = inputs["inp_cnt"] if "inp_cnt" in inputs.keys() else None
        inp_frames = inputs["inp_frames"] if "inp_frames" in inputs.keys() else None
        height = inp_events.shape[2]
        width = inp_events.shape[3]

        # check if new sequence
        path_to = self.store_dir + sequence + "/"
        if not os.path.exists(path_to):
            os.makedirs(path_to)
            os.makedirs(path_to + "events/")
            os.makedirs(path_to + "flow/")
            os.makedirs(path_to + "frames/")
            os.makedirs(path_to + "iwe/")
            os.makedirs(path_to + "brightness/")
            if self.store_file is not None:
                self.store_file.close()
            self.store_file = open(path_to + "timestamps.txt", "w")
            self.img_idx = 0

        # input events
        event_image = np.zeros((height, width))
        inp_events = inp_events.detach()
        inp_events_npy = inp_events.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
        event_image = self.events_to_image(inp_events_npy)
        filename = path_to + "events/%09d.png" % self.img_idx
        cv2.imwrite(filename, event_image * 255)

        # input frames
        if inp_frames is not None:
            inp_frames = inp_frames.detach()
            inp_frames_npy = inp_frames.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            filename = path_to + "frames/%09d.png" % self.img_idx
            cv2.imwrite(filename, inp_frames_npy[:, :, 1])

        # optical flow
        if flow is not None:
            flow = flow.detach()
            flow_npy = flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            flow_npy = self.flow_to_image(flow_npy[:, :, 0], flow_npy[:, :, 1])
            flow_npy = cv2.cvtColor(flow_npy, cv2.COLOR_RGB2BGR)
            filename = path_to + "flow/%09d.png" % self.img_idx
            cv2.imwrite(filename, flow_npy)

        # image of warped events
        if iwe is not None:
            iwe = iwe.detach()
            iwe_npy = iwe.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            iwe_npy = self.events_to_image(iwe_npy)
            filename = path_to + "iwe/%09d.png" % self.img_idx
            cv2.imwrite(filename, iwe_npy * 255)

        # reconstructed brightness
        if brightness is not None:
            brightness = brightness.detach()
            brightness_npy = brightness.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 1))
            intensity_npy = brightness_npy.reshape((height, width, 1))
            intensity_image = self.minmax_norm(intensity_npy)
            filename = path_to + "brightness/%09d.png" % self.img_idx
            cv2.imwrite(filename, intensity_image * 255)

        # store timestamps
        if ts is not None:
            self.store_file.write(str(ts) + "\n")
            self.store_file.flush()

        self.img_idx += 1
        cv2.waitKey(1)

    @staticmethod
    def flow_to_image(flow_x, flow_y):
        """
        Use the optical flow color scheme from the supplementary materials of the paper 'Back to Event
        Basics: Self-Supervised Image Reconstruction for Event Cameras via Photometric Constancy',
        Paredes-Valles et al., CVPR'21.
        :param flow_x: [H x W x 1] horizontal optical flow component
        :param flow_y: [H x W x 1] vertical optical flow component
        :return flow_rgb: [H x W x 3] color-encoded optical flow
        """
        flows = np.stack((flow_x, flow_y), axis=2)
        mag = np.linalg.norm(flows, axis=2)
        min_mag = np.min(mag)
        mag_range = np.max(mag) - min_mag

        ang = np.arctan2(flow_y, flow_x) + np.pi
        ang *= 1.0 / np.pi / 2.0

        hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3])
        hsv[:, :, 0] = ang
        hsv[:, :, 1] = 1.0
        hsv[:, :, 2] = mag - min_mag
        if mag_range != 0.0:
            hsv[:, :, 2] /= mag_range

        flow_rgb = matplotlib.colors.hsv_to_rgb(hsv)
        return (255 * flow_rgb).astype(np.uint8)

    @staticmethod
    def minmax_norm(x):
        """
        Robust min-max normalization.
        :param x: [H x W x 1]
        :return x: [H x W x 1] normalized x
        """
        den = np.percentile(x, 99) - np.percentile(x, 1)
        if den != 0:
            x = (x - np.percentile(x, 1)) / den
        return np.clip(x, 0, 1)

    @staticmethod
    def events_to_image(inp_events, color_scheme="green_red"):
        """
        Visualize the input events.
        :param inp_events: [batch_size x 2 x H x W] per-pixel and per-polarity event count
        :param color_scheme: green_red/gray
        :return event_image: [H x W x 3] color-coded event image
        """
        pos = inp_events[:, :, 0]
        neg = inp_events[:, :, 1]
        pos_max = np.percentile(pos, 99)
        pos_min = np.percentile(pos, 1)
        neg_max = np.percentile(neg, 99)
        neg_min = np.percentile(neg, 1)
        max = pos_max if pos_max > neg_max else neg_max

        if pos_min != max:
            pos = (pos - pos_min) / (max - pos_min)
        if neg_min != max:
            neg = (neg - neg_min) / (max - neg_min)

        pos = np.clip(pos, 0, 1)
        neg = np.clip(neg, 0, 1)

        event_image = np.ones((inp_events.shape[0], inp_events.shape[1]))
        if color_scheme == "gray":
            event_image *= 0.5
            pos *= 0.5
            neg *= -0.5
            event_image += pos + neg

        elif color_scheme == "green_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            event_image *= 0
            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            event_image[:, :, 0][mask_pos] = 0
            event_image[:, :, 1][mask_pos] = pos[mask_pos]
            event_image[:, :, 2][mask_pos * mask_not_neg] = 0
            event_image[:, :, 2][mask_neg] = neg[mask_neg]
            event_image[:, :, 0][mask_neg] = 0
            event_image[:, :, 1][mask_neg * mask_not_pos] = 0

        return event_image
