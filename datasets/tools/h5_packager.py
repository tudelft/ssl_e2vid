"""
Adapted from Monash University https://github.com/TimoStoff/events_contrast_maximization
"""

import h5py
import numpy as np


class H5Packager:
    def __init__(self, output_path):
        print("Creating file in {}".format(output_path))
        self.output_path = output_path

        self.file = h5py.File(output_path, "w")
        self.event_xs = self.file.create_dataset(
            "events/xs",
            (0,),
            dtype=np.dtype(np.int16),
            maxshape=(None,),
            chunks=True,
        )
        self.event_ys = self.file.create_dataset(
            "events/ys",
            (0,),
            dtype=np.dtype(np.int16),
            maxshape=(None,),
            chunks=True,
        )
        self.event_ts = self.file.create_dataset(
            "events/ts",
            (0,),
            dtype=np.dtype(np.float64),
            maxshape=(None,),
            chunks=True,
        )
        self.event_ps = self.file.create_dataset(
            "events/ps",
            (0,),
            dtype=np.dtype(np.bool_),
            maxshape=(None,),
            chunks=True,
        )

    def append(self, dataset, data):
        dataset.resize(dataset.shape[0] + len(data), axis=0)
        if len(data) == 0:
            return
        dataset[-len(data) :] = data[:]

    def package_events(self, xs, ys, ts, ps):
        self.append(self.event_xs, xs)
        self.append(self.event_ys, ys)
        self.append(self.event_ts, ts)
        self.append(self.event_ps, ps)

    def package_image(self, image, timestamp, img_idx):
        image_dset = self.file.create_dataset(
            "images/image{:09d}".format(img_idx),
            data=image,
            dtype=np.dtype(np.uint8),
        )
        image_dset.attrs["size"] = image.shape
        image_dset.attrs["timestamp"] = timestamp
        image_dset.attrs["type"] = "greyscale" if image.shape[-1] == 1 or len(image.shape) == 2 else "color_bgr"

    def add_metadata(
        self,
        num_pos,
        num_neg,
        duration,
        t0,
        tk,
        num_imgs,
        sensor_size,
    ):
        self.file.attrs["num_events"] = num_pos + num_neg
        self.file.attrs["num_pos"] = num_pos
        self.file.attrs["num_neg"] = num_neg
        self.file.attrs["duration"] = tk - t0
        self.file.attrs["t0"] = t0
        self.file.attrs["tk"] = tk
        self.file.attrs["num_imgs"] = num_imgs
        self.file.attrs["sensor_resolution"] = sensor_size
