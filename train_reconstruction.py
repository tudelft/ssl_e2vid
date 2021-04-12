import os
import argparse

import mlflow
import numpy as np
import torch
from torch.optim import *

from configs.parser import YAMLParser
from dataloader.h5 import H5Loader
from loss.flow import EventWarping
from loss.reconstruction import BrightnessConstancy
from models.model import FireFlowNet, EVFlowNet
from models.model import FireNet, E2VID
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
    num_bins = config["data"]["num_bins"]

    # visualization tool
    if config["vis"]["enabled"]:
        vis = Visualization(config)

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

    # loss functions
    loss_function_flow = EventWarping(config, device)
    loss_function_reconstruction = BrightnessConstancy(config, device)

    # reconstruction settings
    model_reconstruction = eval(config["model_reconstruction"]["name"])(
        config["model_reconstruction"].copy(), num_bins
    ).to(device)
    model_reconstruction = load_model(args.prev_model, model_reconstruction, device)
    model_reconstruction.train()

    # optical flow settings
    model_flow = eval(config["model_flow"]["name"])(config["model_flow"].copy(), num_bins).to(device)
    model_flow = load_model(args.prev_model, model_flow, device)
    if config["loss"]["train_flow"]:
        model_flow.train()
    else:
        model_flow.eval()

    # model directory
    path_models = create_model_dir(args.path_models, mlflow.active_run().info.run_id)
    mlflow.log_param("trained_model", path_models)
    config_parser.log_config(path_models)
    config["trained_model"] = path_models
    config_parser.config = config
    config_parser.log_config(path_models)

    # optimizers
    optimizer_reconstruction = eval(config["optimizer"]["name"])(
        model_reconstruction.parameters(), lr=config["optimizer"]["lr"]
    )
    optimizer_flow = eval(config["optimizer"]["name"])(model_flow.parameters(), lr=config["optimizer"]["lr"])
    optimizer_reconstruction.zero_grad()
    optimizer_flow.zero_grad()

    # simulation variables
    seq_length = 0
    loss_reconstruction = 0
    loss_flow = 0
    train_loss_reconstruction = 0
    train_loss_flow = 0
    best_loss_reconstruction = 1.0e6
    best_loss_flow = 1.0e6
    end_train = False

    prev_img = None
    x_reconstruction = None

    # training loop
    data.shuffle()
    while True:
        for inputs in dataloader:

            if data.new_seq:
                seq_length = 0
                data.new_seq = False
                loss_reconstruction = 0
                model_reconstruction.reset_states()
                optimizer_reconstruction.zero_grad()

                prev_img = None
                x_reconstruction = None

            if data.seq_num >= len(data.files):
                mlflow.log_metric(
                    "loss_reconstruction", train_loss_reconstruction / (data.samples + 1), step=data.epoch
                )
                mlflow.log_metric("loss_flow", train_loss_flow / (data.samples + 1), step=data.epoch)

                with torch.no_grad():
                    if train_loss_reconstruction / (data.samples + 1) < best_loss_reconstruction:
                        save_model(path_models, model_reconstruction)
                        best_loss_reconstruction = train_loss_reconstruction / (data.samples + 1)
                    if train_loss_flow / (data.samples + 1) < best_loss_flow:
                        save_model(path_models, model_flow)
                        best_loss_flow = train_loss_flow / (data.samples + 1)

                data.epoch += 1
                data.samples = 0
                train_loss_flow = 0
                train_loss_reconstruction = 0
                data.seq_num = data.seq_num % len(data.files)

                # finish training loop
                if data.epoch == config["loader"]["n_epochs"]:
                    end_train = True

            # forward pass - flow network
            x_flow = model_flow(inputs["inp_voxel"].to(device), inputs["inp_cnt"].to(device))

            # loss and backward pass
            if config["loss"]["train_flow"]:
                loss_flow = loss_function_flow(
                    x_flow["flow"], inputs["inp_list"].to(device), inputs["inp_pol_mask"].to(device)
                )

                train_loss_flow += loss_flow.item()
                loss_flow.backward()
                optimizer_flow.step()
                optimizer_flow.zero_grad()

            if x_reconstruction is not None:

                # reconstruction loss - generative model
                delta_loss_model = loss_function_reconstruction.generative_model(
                    x_flow["flow"][0].detach(), x_reconstruction["image"], inputs
                )
                loss_reconstruction += delta_loss_model
                train_loss_reconstruction += delta_loss_model.item()

                if prev_img is None or "Pause" not in data.batch_augmentation or not data.batch_augmentation["Pause"]:

                    # reconstruction loss - regularization
                    delta_loss_reg = loss_function_reconstruction.regularization(x_reconstruction["image"])
                    loss_reconstruction += delta_loss_reg
                    train_loss_reconstruction += delta_loss_reg.item()

                    # update previous image
                    prev_img = x_reconstruction["image"].detach().clone()

            # forward pass - reconstruction network
            x_reconstruction = model_reconstruction(inputs["inp_voxel"].to(device))
            data.tc_idx += 1

            # reconstruction loss - temporal constancy
            if data.tc_idx >= config["loss"]["reconstruction_tc_idx_threshold"]:
                delta_loss_tc = loss_function_reconstruction.temporal_consistency(
                    x_flow["flow"][0].detach(), prev_img, x_reconstruction["image"]
                )
                loss_reconstruction += delta_loss_tc
                train_loss_reconstruction += delta_loss_tc.item()

            # update sequence length
            seq_length += 1

            # visualize
            with torch.no_grad():
                if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                    vis.update(inputs, x_flow["flow"][-1], None, x_reconstruction["image"])

            # reconstruction backward pass
            if seq_length == config["loss"]["reconstruction_unroll"]:

                if loss_reconstruction != 0:
                    loss_reconstruction.backward()
                    optimizer_reconstruction.step()
                    optimizer_reconstruction.zero_grad()

                seq_length = 0
                x_reconstruction = None
                loss_reconstruction = 0

                # detach states
                model_reconstruction.detach_states()

            # print training info
            if config["vis"]["verbose"]:
                print(
                    "Train Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)] Flow loss: {:.6f}, Brightness loss: {:.6f}".format(
                        data.epoch,
                        data.seq_num,
                        len(data.files),
                        int(100 * data.seq_num / len(data.files)),
                        train_loss_flow / (data.samples + 1),
                        train_loss_reconstruction / (data.samples + 1),
                    ),
                    end="\r",
                )

            # update number of samples seen by the network
            data.samples += config["loader"]["batch_size"]

        if end_train:
            break

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_reconstruction.yml",
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
