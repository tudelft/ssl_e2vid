import os
import datetime

import torch


def load_model(model_dir, model, device):
    """
    Load model from file.
    :param model_dir: model directory
    :param model: instance of the model class to be loaded
    :param device: model device
    :return loaded model
    """

    if os.path.isfile(model_dir):
        model_loaded = torch.load(model_dir, map_location=device)
        if "state_dict" in model_loaded.keys():
            model_loaded = model_loaded["state_dict"]
        model.load_state_dict(model_loaded)
        print("Model restored from " + model_dir + "\n")

    elif os.path.isdir(model_dir):
        model_name = model_dir + model.__class__.__name__

        extensions = [".pt", ".pth.tar", ".pwf", "_weights_min.pwf"]  # backwards compatibility
        for ext in extensions:
            if os.path.isfile(model_name + ext):
                model_name += ext
                break

        if os.path.isfile(model_name):
            model_loaded = torch.load(model_name, map_location=device)
            if "state_dict" in model_loaded.keys():
                model_loaded = model_loaded["state_dict"]
            model.load_state_dict(model_loaded)
            print("Model restored from " + model_name + "\n")
        else:
            print("No model found at" + model_name + "\n")

    return model


def create_model_dir(path_models, runid):
    """
    Create directory for storing model parameters.
    :param path_models: path in which the model should be stored
    :param runid: MLFlow's unique ID of the model
    :return path to generated model directory
    """

    now = datetime.datetime.now()

    path_models += "model_"
    path_models += "%02d%02d%04d" % (now.day, now.month, now.year)
    path_models += "_%02d%02d%02d_" % (now.hour, now.minute, now.second)
    path_models += runid  # mlflow run ID
    path_models += "/"
    if not os.path.exists(path_models):
        os.makedirs(path_models)
    print("Weights stored at " + path_models + "\n")
    return path_models


def save_model(path_models, model):
    """
    Overwrite previously saved model with new parameters.
    :param path_models: model directory
    :param model: instance of the model class to be saved
    """

    os.system("rm -rf " + path_models + model.__class__.__name__ + ".pt")
    model_name = path_models + model.__class__.__name__ + ".pt"
    torch.save(model.state_dict(), model_name)
