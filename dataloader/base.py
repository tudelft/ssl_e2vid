from abc import abstractmethod

import numpy as np
import random
import torch

from .encodings import events_to_voxel, events_to_channels, events_to_mask, get_hot_event_mask


class BaseDataLoader(torch.utils.data.Dataset):
    """
    Base class for dataloader.
    """

    def __init__(self, config, num_bins):
        self.config = config
        self.epoch = 0
        self.seq_num = 0
        self.samples = 0
        self.new_seq = False
        self.tc_idx = 0
        self.num_bins = num_bins

        # batch-specific data augmentation mechanisms
        self.batch_augmentation = {}
        for mechanism in self.config["loader"]["augment"]:
            if mechanism != "Pause":
                self.batch_augmentation[mechanism] = [False for i in range(self.config["loader"]["batch_size"])]
            else:
                self.batch_augmentation[mechanism] = False  # shared among batch elements

        for i, mechanism in enumerate(self.config["loader"]["augment"]):
            if mechanism != "Pause":
                for batch in range(self.config["loader"]["batch_size"]):
                    if np.random.random() < self.config["loader"]["augment_prob"][i]:
                        self.batch_augmentation[mechanism][batch] = True

        # hot pixels
        if self.config["hot_filter"]["enabled"]:
            self.hot_idx = [0 for i in range(self.config["loader"]["batch_size"])]
            self.hot_events = [
                torch.zeros(self.config["loader"]["resolution"]) for i in range(self.config["loader"]["batch_size"])
            ]

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def get_events(self, history):
        raise NotImplementedError

    def reset_sequence(self, batch):
        """
        Reset sequence-specific variables.
        :param batch: batch index
        """

        self.tc_idx = 0
        self.seq_num += 1
        if self.config["hot_filter"]["enabled"]:
            self.hot_idx[batch] = 0
            self.hot_events[batch] = torch.zeros(self.config["loader"]["resolution"])

        for i, mechanism in enumerate(self.config["loader"]["augment"]):
            if mechanism != "Pause":
                if np.random.random() < self.config["loader"]["augment_prob"][i]:
                    self.batch_augmentation[mechanism][batch] = True
                else:
                    self.batch_augmentation[mechanism][batch] = False
            else:
                self.batch_augmentation[mechanism] = False

    @staticmethod
    def event_formatting(xs, ys, ts, ps):
        """
        Reset sequence-specific variables.
        :param xs: [N] numpy array with event x location
        :param ys: [N] numpy array with event y location
        :param ts: [N] numpy array with event timestamp
        :param ps: [N] numpy array with event polarity ([-1, 1])
        :return xs: [N] tensor with event x location
        :return ys: [N] tensor with event y location
        :return ts: [N] tensor with normalized event timestamp
        :return ps: [N] tensor with event polarity ([-1, 1])
        """

        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(ys.astype(np.float32))
        ts = torch.from_numpy(ts.astype(np.float32))
        ps = torch.from_numpy(ps.astype(np.float32)) * 2 - 1
        ts = (ts - ts[0]) / (ts[-1] - ts[0])
        return xs, ys, ts, ps

    def augment_events(self, xs, ys, ps, batch):
        """
        Augment event sequence with horizontal, vertical, and polarity flips, and
        artificial event pauses.
        :return xs: [N] tensor with event x location
        :return ys: [N] tensor with event y location
        :return ps: [N] tensor with event polarity ([-1, 1])
        :param batch: batch index
        :return xs: [N] tensor with augmented event x location
        :return ys: [N] tensor with augmented event y location
        :return ps: [N] tensor with augmented event polarity ([-1, 1])
        """

        for i, mechanism in enumerate(self.config["loader"]["augment"]):

            if mechanism == "Horizontal":
                if self.batch_augmentation["Horizontal"][batch]:
                    xs = self.config["loader"]["resolution"][1] - 1 - xs

            elif mechanism == "Vertical":
                if self.batch_augmentation["Vertical"][batch]:
                    ys = self.config["loader"]["resolution"][0] - 1 - ys

            elif mechanism == "Polarity":
                if self.batch_augmentation["Polarity"][batch]:
                    ps *= -1

            # shared among batch elements
            elif (
                batch == 0
                and mechanism == "Pause"
                and self.tc_idx > self.config["loss"]["reconstruction_tc_idx_threshold"]
            ):
                if not self.batch_augmentation["Pause"]:
                    if np.random.random() < self.config["loader"]["augment_prob"][i][0]:
                        self.batch_augmentation["Pause"] = True
                else:
                    if np.random.random() < self.config["loader"]["augment_prob"][i][1]:
                        self.batch_augmentation["Pause"] = False

        return xs, ys, ps

    def augment_frames(self, img, batch):
        """
        Augment APS frame with horizontal and vertical flips.
        :param img: [H x W] numpy array with APS intensity
        :param batch: batch index
        :return img: [H x W] augmented numpy array with APS intensity
        """
        if "Horizontal" in self.batch_augmentation:
            if self.batch_augmentation["Horizontal"][batch]:
                img = np.flip(img, 1)
        if "Vertical" in self.batch_augmentation:
            if self.batch_augmentation["Vertical"][batch]:
                img = np.flip(img, 0)
        return img

    def create_cnt_encoding(self, xs, ys, ts, ps):
        """
        Creates a per-pixel and per-polarity event count representation.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [2 x H x W] event representation
        """

        return events_to_channels(xs, ys, ps, sensor_size=self.config["loader"]["resolution"])

    def create_voxel_encoding(self, xs, ys, ts, ps):
        """
        Creates a spatiotemporal voxel grid tensor representation with a certain number of bins,
        as described in Section 3.1 of the paper 'Unsupervised Event-based Learning of Optical Flow,
        Depth, and Egomotion', Zhu et al., CVPR'19..
        Events are distributed to the spatiotemporal closest bins through bilinear interpolation.
        Positive events are added as +1, while negative as -1.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [B x H x W] event representation
        """

        return events_to_voxel(
            xs,
            ys,
            ts,
            ps,
            self.num_bins,
            sensor_size=self.config["loader"]["resolution"],
        )

    @staticmethod
    def create_list_encoding(xs, ys, ts, ps):
        """
        Creates a four channel tensor with all the events in the input partition.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 4] event representation
        """

        return torch.stack([ts, ys, xs, ps])

    @staticmethod
    def create_polarity_mask(ps):
        """
        Creates a two channel tensor that acts as a mask for the input event list.
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 2] event representation
        """

        inp_pol_mask = torch.stack([ps, ps])
        inp_pol_mask[0, :][inp_pol_mask[0, :] < 0] = 0
        inp_pol_mask[1, :][inp_pol_mask[1, :] > 0] = 0
        inp_pol_mask[1, :] *= -1
        return inp_pol_mask

    def create_hot_mask(self, xs, ys, ps, batch):
        """
        Creates a one channel tensor that can act as mask to remove pixel with high event rate.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [H x W] binary mask
        """

        hot_update = events_to_mask(xs, ys, ps, sensor_size=self.hot_events[batch].shape)
        self.hot_events[batch] += hot_update
        self.hot_idx[batch] += 1
        event_rate = self.hot_events[batch] / self.hot_idx[batch]
        return get_hot_event_mask(
            event_rate,
            self.hot_idx[batch],
            max_px=self.config["hot_filter"]["max_px"],
            min_obvs=self.config["hot_filter"]["min_obvs"],
            max_rate=self.config["hot_filter"]["max_rate"],
        )

    def __len__(self):
        return 1000  # not used

    @staticmethod
    def custom_collate(batch):
        """
        Collects the different event representations and stores them together in a dictionary.
        """

        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = []
        for entry in batch:
            for key in entry.keys():
                batch_dict[key].append(entry[key])
        for key in batch_dict.keys():
            item = torch.stack(batch_dict[key])
            if len(item.shape) == 3:
                item = item.transpose(2, 1)
            batch_dict[key] = item
        return batch_dict

    def shuffle(self, flag=True):
        """
        Shuffles the training data.
        """

        if flag:
            random.shuffle(self.files)
