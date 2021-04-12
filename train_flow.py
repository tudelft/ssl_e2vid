import os
import argparse

import mlflow
import numpy as np
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import EventWarping
from models.model import FireFlowNet, EVFlowNet
from utils.utils import load_model, create_model_dir, save_model
from utils.visualization import Visualization


def train(args, config_parser):
    if not os.path.exists(args.path_models):
        os.makedirs(args.path_models)

    # configs
    config = config_parser.config
    config["vis"]["bars"] = False

    # log config
    mlflow.set_experiment(config["experiment"])
    mlflow.start_run()
    mlflow.log_params(config)
    mlflow.log_param("prev_model", args.prev_model)
    config["prev_model"] = args.prev_model

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # visualization tool
    if config["vis"]["enabled"]:
        vis = Visualization(config)

    # loss functions
    loss_function = EventWarping(config, device)

    # optical flow settings
    num_bins = config["data"]["num_bins"]
    model = eval(config["model_flow"]["name"])(config["model_flow"].copy(), num_bins).to(device)
    model = load_model(args.prev_model, model, device)
    model.train()

    # model directory
    path_models = create_model_dir(args.path_models, mlflow.active_run().info.run_id)
    mlflow.log_param("trained_model", path_models)
    config["trained_model"] = path_models
    config_parser.config = config
    config_parser.log_config(path_models)

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

    # optimizers
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    optimizer.zero_grad()

    # simulation variables
    loss = 0
    train_loss = 0
    best_loss = 1.0e6
    end_train = False

    # training loop
    data.shuffle()
    while True:
        for inputs in dataloader:

            # check new epoch
            if data.seq_num >= len(data.files):
                mlflow.log_metric("loss_flow", train_loss / (data.samples + 1), step=data.epoch)

                with torch.no_grad():
                    if train_loss / (data.samples + 1) < best_loss:
                        save_model(path_models, model)
                        best_loss = train_loss / (data.samples + 1)

                data.epoch += 1
                data.samples = 0
                train_loss = 0
                data.seq_num = data.seq_num % len(data.files)

                # finish training loop
                if data.epoch == config["loader"]["n_epochs"]:
                    end_train = True

            # forward pass
            x = model(inputs["inp_voxel"].to(device), inputs["inp_cnt"].to(device))

            # loss and backward pass
            loss = loss_function(x["flow"], inputs["inp_list"].to(device), inputs["inp_pol_mask"].to(device))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # print training info
            if config["vis"]["verbose"]:
                print(
                    "Train Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)] loss: {:.6f}".format(
                        data.epoch,
                        data.seq_num,
                        len(data.files),
                        int(100 * data.seq_num / len(data.files)),
                        train_loss / (data.samples + 1),
                    ),
                    end="\r",
                )

            # visualize
            with torch.no_grad():
                if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                    vis.update(inputs, x["flow"][-1], None, None)

            # update number of samples seen by the network
            data.samples += config["loader"]["batch_size"]

        if end_train:
            break

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_flow.yml",
        help="training configuration",
    )
    parser.add_argument(
        "--path_models",
        default="trained_models/",
        help="location of trained models",
    )
    parser.add_argument(
        "--prev_model",
        default="",
        help="pre-trained model to use as starting point",
    )
    args = parser.parse_args()

    # launch training
    train(args, YAMLParser(args.config))
