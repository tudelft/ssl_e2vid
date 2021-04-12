import argparse

import numpy as np
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from models.model import FireFlowNet, EVFlowNet
from utils.iwe import deblur_events, compute_pol_iwe
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

    # visualization tool
    if config["vis"]["enabled"] or config["vis"]["store"]:
        vis = Visualization(config, eval_id=eval_id)

    # optical flow settings
    num_bins = config["data"]["num_bins"]
    flow_scaling = config["model_flow"]["flow_scaling"]
    model = eval(config["model_flow"]["name"])(config["model_flow"], num_bins).to(device)
    model = load_model(config["trained_model"], model, device)
    model.eval()

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

    # inference loop
    end_test = False
    with torch.no_grad():
        while True:
            for inputs in dataloader:

                # finish inference loop
                if data.seq_num >= len(data.files):
                    end_test = True
                    break

                # forward pass
                x = model(inputs["inp_voxel"].to(device), inputs["inp_cnt"].to(device))

                # image of warped events
                iwe = compute_pol_iwe(
                    x["flow"][-1],
                    inputs["inp_list"].to(device),
                    config["loader"]["resolution"],
                    inputs["inp_pol_mask"][:, :, 0:1].to(device),
                    inputs["inp_pol_mask"][:, :, 1:2].to(device),
                    flow_scaling=flow_scaling,
                    round_idx=False,
                )

                # visualize
                for bar in data.open_files_bar:
                    bar.next()
                if config["vis"]["enabled"]:
                    vis.update(inputs, x["flow"][-1], iwe, None)
                if config["vis"]["store"]:
                    sequence = data.files[data.batch_idx[0] % len(data.files)].split("/")[-1].split(".")[0]
                    vis.store(inputs, x["flow"][-1], iwe, None, sequence, ts=data.last_proc_timestamp)

            if end_test:
                break

    for bar in data.open_files_bar:
        bar.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trained_model", help="model to be evaluated")
    parser.add_argument(
        "--config",
        default="configs/eval_flow.yml",
        help="config file, overwrites model settings",
    )
    args = parser.parse_args()

    # launch testing
    test(args, YAMLParser(args.config))
