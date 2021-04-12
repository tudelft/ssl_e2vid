import os
import glob
import h5py
import argparse
import numpy as np
from ast import literal_eval

from h5_packager import H5Packager


class Frames:
    def __init__(self):
        self.ts = []
        self.names = []

    def __call__(self, name, h5obj):
        if hasattr(h5obj, "dtype") and name not in self.names:
            self.names += [name]
            self.ts += [h5obj.attrs["timestamp"]]

    def get_frames(self, file, t0, t1, crop, res):
        idx0 = binary_search_array(self.ts, t0)
        idx1 = binary_search_array(self.ts, t1)

        imgs = []
        for i in range(idx0, idx1):
            imgs.append(file["images"]["image{:09d}".format(i)][crop[0] : crop[0] + res[0], crop[1] : crop[1] + res[1]])

        return imgs, idx0, idx1


def binary_search_array(array, x, l=None, r=None, side="left"):
    """
    Binary search through a sorted array.
    """

    l = 0 if l is None else l
    r = len(array) - 1 if r is None else r
    mid = l + (r - l) // 2

    if l > r:
        return l if side == "left" else r

    if array[mid] == x:
        return mid
    elif x < array[mid]:
        return binary_search_array(array, x, l=l, r=mid - 1)
    else:
        return binary_search_array(array, x, l=mid + 1, r=r)


def find_ts_index(file, timestamp):
    idx = binary_search_array(file["events/ts"], timestamp)
    return idx


def get_events(file, idx0, idx1):
    xs = file["events/xs"][idx0:idx1]
    ys = file["events/ys"][idx0:idx1]
    ts = file["events/ts"][idx0:idx1]
    ps = file["events/ps"][idx0:idx1] * 2 - 1
    return xs, ys, ts, ps


def random_crop(original_res, output_res):
    h = np.random.randint(0, high=original_res[0] - 1 - output_res[0])
    w = np.random.randint(0, high=original_res[1] - 1 - output_res[1])
    return h, w


if __name__ == "__main__":
    """
    Tool for generating a training dataset out of a set of specified group of H5 datasets.
    The original sequences are cropped both in time and space.
    The resulting training sequences contain the raw images if available.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="directory of datasets to be used")
    parser.add_argument(
        "--output_dir",
        default="/tmp/training",
        help="output directory containing the resulting training sequences",
    )
    parser.add_argument(
        "--time_length",
        default=2,
        help="maximum duration, in seconds, of the training sequences",
        type=float,
    )
    parser.add_argument(
        "--original_res",
        default="(180, 240)",
        help="resolution of the original sequences, HxW",
    )
    parser.add_argument(
        "--output_res",
        default="(128,128)",
        help="resolution of the resulting sequences, HxW",
    )
    parser.add_argument(
        "--with_images",
        default=False,
        help="whether or not the resulting dataset should contain images",
    )
    args = parser.parse_args()
    original_res = literal_eval(args.original_res)
    output_res = literal_eval(args.output_res)

    print("Data will be extracted in folder: {}".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    path_from = []
    for root, dirs, files in os.walk(args.path):
        for file in files:
            if file.endswith(".h5"):
                path_from.append(os.path.join(root, file))

    # process dataset
    for path in path_from:
        hf = h5py.File(path, "r")
        print("Processing:", path)
        filename = path.split("/")[-1].split(".")[0]

        # load image data
        if args.with_images:
            frames = Frames()
            hf["images"].visititems(frames)

        # get subsequence random crop params
        crop = random_crop(original_res, output_res)

        # start reading sequence
        t = hf["events/ts"][0]  # s
        idx0 = find_ts_index(hf, t)

        sequence_id = 0
        while True:
            idx1 = find_ts_index(hf, t + args.time_length)

            # events in temporal window
            xs, ys, ts, ps = get_events(hf, idx0, idx1)
            if len(xs) == 0:
                break

            # events in spatial window
            x_out = np.argwhere(xs < crop[1])
            x_out = np.concatenate((x_out, np.argwhere(xs >= crop[1] + output_res[1])), axis=0)
            xs = np.delete(xs, x_out)
            ys = np.delete(ys, x_out)
            ts = np.delete(ts, x_out)
            ps = np.delete(ps, x_out)

            y_out = np.argwhere(ys < crop[0])
            y_out = np.concatenate((y_out, np.argwhere(ys >= crop[0] + output_res[0])), axis=0)
            xs = np.delete(xs, y_out)
            ys = np.delete(ys, y_out)
            ts = np.delete(ts, y_out)
            ps = np.delete(ps, y_out)
            ps[ps < 0] = 0

            xs -= crop[1]
            ys -= crop[0]

            # images in temporal window
            cropped_frames = []
            if args.with_images:
                cropped_frames, frame_idx0, frame_idx1 = frames.get_frames(
                    hf, t, t + args.time_length, crop, output_res
                )

            # store subsequence
            ep = H5Packager(args.output_dir + filename + "_" + str(sequence_id) + ".h5")
            ep.package_events(xs.tolist(), ys.tolist(), ts.tolist(), ps.tolist())

            img_cnt = 0
            if args.with_images:
                for i in range(frame_idx0, frame_idx1):
                    ep.package_image(cropped_frames[img_cnt], frames.ts[i], img_cnt)
                    img_cnt += 1

            ep.add_metadata(
                len(ps[ps > 0]),
                len(ps[ps < 0]),
                ts[-1] - ts[0],
                ts[0],
                ts[-1],
                img_cnt,
                output_res,
            )

            t += args.time_length
            idx0 = idx1
            sequence_id += 1

        hf.close()
        print("")
