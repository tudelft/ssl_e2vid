import os
import h5py
import numpy as np

import torch
import torch.utils.data as data

from .base import BaseDataLoader
from .utils import ProgressBar

from .encodings import binary_search_array


class Frames:
    """
    Utility class for reading the APS frames encoded in the HDF5 files.
    """

    def __init__(self):
        self.ts = []
        self.names = []

    def __call__(self, name, h5obj):
        if hasattr(h5obj, "dtype") and name not in self.names:
            self.names += [name]
            self.ts += [h5obj.attrs["timestamp"]]

    def get_frames(self, file, t0, t1, crop, res):
        """
        Get all the APS frames in between two timestamps.
        :param file: file to read from
        :param t0: start time
        :param t1: end time
        :param crop: top-left corner of the patch to be cropped
        :param res: resolution of the patch to be cropped
        :return imgs: list of [H x W] APS images
        :return idx0: index of the first frame
        :return idx1: index of the last frame
        """

        idx0 = binary_search_array(self.ts, t0)
        idx1 = binary_search_array(self.ts, t1)

        imgs = []
        for i in range(idx0, idx1):
            imgs.append(file["images"]["image{:09d}".format(i)][crop[0] : crop[0] + res[0], crop[1] : crop[1] + res[1]])

        return imgs, idx0, idx1


class H5Loader(BaseDataLoader):
    def __init__(self, config, num_bins):
        super().__init__(config, num_bins)
        self.last_proc_timestamp = 0

        # "memory" that goes from forward pass to the next
        self.batch_idx = [i for i in range(self.config["loader"]["batch_size"])]  # event sequence
        self.batch_row = [0 for i in range(self.config["loader"]["batch_size"])]  # event_idx / time_idx / frame_idx
        self.batch_t0 = [None for i in range(self.config["loader"]["batch_size"])]

        # input event sequences
        self.files = []
        for root, dirs, files in os.walk(config["data"]["path"]):
            for file in files:
                if file.endswith(".h5"):
                    self.files.append(os.path.join(root, file))

        # open first files
        self.open_files = []
        for batch in range(self.config["loader"]["batch_size"]):
            self.open_files.append(h5py.File(self.files[batch], "r"))

        # load frames from open files
        self.open_files_frames = []
        if self.config["data"]["mode"] == "frames":
            for batch in range(self.config["loader"]["batch_size"]):
                frames = Frames()
                self.open_files[batch]["images"].visititems(frames)
                self.open_files_frames.append(frames)

        # progress bars
        if self.config["vis"]["bars"]:
            self.open_files_bar = []
            for batch in range(self.config["loader"]["batch_size"]):
                max_iters = self.get_iters(batch)
                self.open_files_bar.append(ProgressBar(self.files[batch].split("/")[-1], max=max_iters))

    def get_iters(self, batch):
        """
        Compute the number of forward passes given a sequence and an input mode and window.
        """

        if self.config["data"]["mode"] == "events":
            max_iters = len(self.open_files[batch]["events/xs"])
        elif self.config["data"]["mode"] == "time":
            max_iters = self.open_files[batch].attrs["duration"]
        elif self.config["data"]["mode"] == "frames":
            max_iters = len(self.open_files_frames[batch].ts) - 1
        else:
            print("DataLoader error: Unknown mode.")
            raise AttributeError

        return max_iters // self.config["data"]["window"]

    def get_events(self, file, idx0, idx1):
        """
        Get all the events in between two indices.
        :param file: file to read from
        :param idx0: start index
        :param idx1: end index
        :return xs: [N] numpy array with event x location
        :return ys: [N] numpy array with event y location
        :return ts: [N] numpy array with event timestamp
        :return ps: [N] numpy array with event polarity ([-1, 1])
        """

        xs = file["events/xs"][idx0:idx1]
        ys = file["events/ys"][idx0:idx1]
        ts = file["events/ts"][idx0:idx1]
        ps = file["events/ps"][idx0:idx1]
        ts -= file.attrs["t0"]  # sequence starting at t0 = 0
        if ts.shape[0] > 0:
            self.last_proc_timestamp = ts[-1]
        ts *= 1.0e6  # us
        return xs, ys, ts, ps

    def get_event_index(self, batch, window=0):
        """
        Get all the event indices to be used for reading.
        :param batch: batch index
        :param window: input window
        :return event_idx: event index
        """

        event_idx = None
        if self.config["data"]["mode"] == "events":
            event_idx = self.batch_row[batch] + window
        elif self.config["data"]["mode"] == "time":
            event_idx = self.find_ts_index(
                self.open_files[batch], self.batch_row[batch] + self.open_files[batch].attrs["t0"] + window
            )
        elif self.config["data"]["mode"] == "frames":
            event_idx = self.find_ts_index(
                self.open_files[batch], self.open_files_frames[batch].ts[self.batch_row[batch] + window]
            )
        else:
            print("DataLoader error: Unknown mode.")
            raise AttributeError
        return event_idx

    def find_ts_index(self, file, timestamp):
        """
        Find closest event index for a given timestamp through binary search.
        """

        return binary_search_array(file["events/ts"], timestamp)

    def __getitem__(self, index):
        while True:
            batch = index % self.config["loader"]["batch_size"]

            # trigger sequence change
            restart = False
            if self.config["data"]["mode"] == "frames" and self.batch_row[batch] + self.config["data"]["window"] >= len(
                self.open_files_frames[batch].ts
            ):
                restart = True

            # load events
            xs = np.zeros((0))
            ys = np.zeros((0))
            ts = np.zeros((0))
            ps = np.zeros((0))
            if not restart:
                idx0 = self.get_event_index(batch)
                idx1 = self.get_event_index(batch, window=self.config["data"]["window"])
                xs, ys, ts, ps = self.get_events(self.open_files[batch], idx0, idx1)

            # trigger sequence change
            if (self.config["data"]["mode"] == "events" and xs.shape[0] < self.config["data"]["window"]) or xs.shape[
                0
            ] <= 10:
                restart = True

            # reset sequence if not enough input events
            if restart:
                self.new_seq = True
                self.reset_sequence(batch)
                self.batch_row[batch] = 0
                self.batch_idx[batch] = max(self.batch_idx) + 1
                self.batch_t0[batch] = None

                self.open_files[batch].close()
                self.open_files[batch] = h5py.File(self.files[self.batch_idx[batch] % len(self.files)], "r")

                if self.config["data"]["mode"] == "frames":
                    frames = Frames()
                    self.open_files[batch]["images"].visititems(frames)
                    self.open_files_frames[batch] = frames
                if self.config["vis"]["bars"]:
                    self.open_files_bar[batch].finish()
                    max_iters = self.get_iters(batch)
                    self.open_files_bar[batch] = ProgressBar(
                        self.files[self.batch_idx[batch] % len(self.files)].split("/")[-1], max=max_iters
                    )

                continue

            # timestamp normalization
            if self.batch_t0[batch] is None:
                self.batch_t0[batch] = ts[0]
            ts -= self.batch_t0[batch]
            xs, ys, ts, ps = self.event_formatting(xs, ys, ts, ps)

            # data augmentation
            xs, ys, ps = self.augment_events(xs, ys, ps, batch)

            # artificial pauses to the event stream
            if "Pause" in self.config["loader"]["augment"]:
                if self.batch_augmentation["Pause"]:
                    xs = torch.from_numpy(np.empty([0]).astype(np.float32))
                    ys = torch.from_numpy(np.empty([0]).astype(np.float32))
                    ts = torch.from_numpy(np.empty([0]).astype(np.float32))
                    ps = torch.from_numpy(np.empty([0]).astype(np.float32))

            # events to tensors
            inp_cnt = self.create_cnt_encoding(xs, ys, ts, ps)
            inp_voxel = self.create_voxel_encoding(xs, ys, ts, ps)
            inp_list = self.create_list_encoding(xs, ys, ts, ps)
            inp_pol_mask = self.create_polarity_mask(ps)

            # hot pixel removal
            if self.config["hot_filter"]["enabled"]:
                hot_mask = self.create_hot_mask(xs, ys, ps, batch)
                hot_mask_voxel = torch.stack([hot_mask] * self.num_bins, axis=2).permute(2, 0, 1)
                hot_mask_cnt = torch.stack([hot_mask] * 2, axis=2).permute(2, 0, 1)
                inp_voxel = inp_voxel * hot_mask_voxel
                inp_cnt = inp_cnt * hot_mask_cnt

            # load frames when required
            if self.config["data"]["mode"] == "frames":
                inp_frames = np.zeros(
                    (
                        2,
                        self.config["loader"]["resolution"][0],
                        self.config["loader"]["resolution"][1],
                    )
                )
                img0 = self.open_files[batch]["images"][self.open_files_frames[batch].names[self.batch_row[batch]]][:]
                img1 = self.open_files[batch]["images"][
                    self.open_files_frames[batch].names[self.batch_row[batch] + self.config["data"]["window"]]
                ][:]
                inp_frames[0, :, :] = self.augment_frames(img0, batch)
                inp_frames[1, :, :] = self.augment_frames(img1, batch)
                inp_frames = torch.from_numpy(inp_frames.astype(np.uint8))

            # update window if not in pause mode
            if "Pause" in self.config["loader"]["augment"]:
                if not self.batch_augmentation["Pause"]:
                    self.batch_row[batch] += self.config["data"]["window"]
            else:
                self.batch_row[batch] += self.config["data"]["window"]

            # break while loop if everything went well
            break

        # prepare output
        output = {}
        output["inp_cnt"] = inp_cnt
        output["inp_voxel"] = inp_voxel
        output["inp_list"] = inp_list
        output["inp_pol_mask"] = inp_pol_mask
        if self.config["data"]["mode"] == "frames":
            output["inp_frames"] = inp_frames

        return output
