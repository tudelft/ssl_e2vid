import os
import sys
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from models.model import FireFlowNet, EVFlowNet
from models.model import FireNet, E2VID
from utils.utils import load_model
from utils.visualization import Visualization


def test(args, config_parser):
    config = config_parser.merge_configs(args.trained_model)
    config["loader"]["batch_size"] = 1
    config["vis"]["bars"] = True

    # store validation settings
    eval_id = config_parser.log_eval_config(config)

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs
    num_bins = config["data"]["num_bins"]

    # visualization tool
    if config["vis"]["enabled"] or config["vis"]["store"]:
        vis = Visualization(config, eval_id=eval_id)

    # data loader
    data = H5Loader(config, num_bins)
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs
    )

    # reconstruction settings
    model_reconstruction = eval(config["model_reconstruction"]["name"])(config["model_reconstruction"], num_bins).to(
        device
    )
    model_reconstruction = load_model(config["trained_model"], model_reconstruction, device)
    model_reconstruction.eval()

    # optical flow settings
    flow_eval = config["model_flow"]["eval"]
    if flow_eval:
        model_flow = eval(config["model_flow"]["name"])(config["model_flow"], num_bins).to(device)
        model_flow = load_model(config["trained_model"], model_flow, device)
        model_flow.eval()

    # inference loop
    x_flow = {}
    x_flow["flow"] = [None]
    end_test = False
    with torch.no_grad():
        while True:
            for inputs in dataloader:

                # reset states
                if data.new_seq:
                    data.new_seq = False
                    model_reconstruction.reset_states()

                # finish inference loop
                if data.seq_num >= len(data.files):
                    end_test = True
                    break

                # flow - forward pass
                if flow_eval:
                    x_flow = model_flow(inputs["inp_voxel"].to(device), inputs["inp_cnt"].to(device))

                # reconstruction - forward pass
                x_reconstruction = model_reconstruction(inputs["inp_voxel"].to(device))

                # visualize
                if config["vis"]["bars"]:
                    for bar in data.open_files_bar:
                        bar.next()
                if config["vis"]["enabled"]:
                    vis.update(inputs, x_flow["flow"][-1], None, x_reconstruction["image"])
                if config["vis"]["store"]:
                    sequence = data.files[data.batch_idx[0] % len(data.files)].split("/")[-1].split(".")[0]
                    vis.store(
                        inputs,
                        x_flow["flow"][-1],
                        None,
                        x_reconstruction["image"],
                        sequence,
                        ts=data.last_proc_timestamp,
                    )

            if end_test:
                break

    if config["vis"]["bars"]:
        for bar in data.open_files_bar:
            bar.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trained_model", help="model to be evaluated")
    parser.add_argument(
        "--config",
        default="configs/eval_reconstruction.yml",
        help="config file, overwrites model settings",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))
