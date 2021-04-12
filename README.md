# Back to Event Basics: SSL of Image Reconstruction for Event Cameras

Minimal code for Back to Event Basics: Self-Supervised Learning of Image Reconstruction for Event Cameras via Photometric Constancy, CVPR'21.

## Usage

This project uses Python >= 3.7.3. After setting up your virtual environment, please install the required python libraries through:

```
pip install -r requirements.txt
```

Code is formatted with Black (PEP8) using a pre-commit hook. To configure it, run:

```
pre-commit install
```

### Data format

Similarly to researchers from [Monash University](https://github.com/TimoStoff/events_contrast_maximization/tree/d6241dc90ec4dc2b4cffbb331a2389ff179bf7ab), this project processes events through the HDF5 data format. Details about the structure of these files can be found in `datasets/tools/`.

## Inference

Download our pre-trained models from [here](https://surfdrive.surf.nl/files/index.php/s/sv36q8ZTqVZuWl9). 

Our HDF5 version of sequences from the Event Camera Dataset can also be downloaded from [here](https://surfdrive.surf.nl/files/index.php/s/sv36q8ZTqVZuWl9) for evaluation purposes.

To estimate optical flow from the input events:

```
python eval_flow.py <path_to_model_dir>
```

<img src=".readme/flow.gif" width="880" height="220" />

&nbsp;

To perform image reconstruction from the input events:

```
python eval_reconstruction.py <path_to_model_dir>
```

<img src=".readme/reconstruction.gif" width="586.6666666666666" height="220" />

&nbsp;

In `configs/`, you can find the configuration files associated to these scripts and vary the inference settings (e.g., number of input events, dataset).

## Training

Our framework can be trained using any event camera dataset. However, if you are interested in using our training data, you can download it from [here](https://surfdrive.surf.nl/files/index.php/s/sv36q8ZTqVZuWl9). The datasets are expected at `datasets/data/`, but this location can be modified in the configuration files.

To train an image reconstruction and optical flow model, you need to adapt the training settings in `configs/train_reconstruction.yml`. Here, you can choose the training dataset, the number of input events, the neural networks to be used (EV-FlowNet or FireFlowNet for optical flow; E2VID or FireNet for image reconstruction), the number of epochs, the optimizer and learning rate, etc. To start the training from scratch, run:

```
python train_reconstruction.py
```

Alternatively, if you have a model that you would like to keep training from, you can use

```
python train_reconstruction.py --prev_model <path_to_model_dir>
```

This is handy if, for instance, you just want to train the image reconstruction model and use a pre-trained optical flow network. For this, you can set `train_flow: False` in `configs/train_reconstruction.yml`, and run:

```
python train_reconstruction.py --prev_model <path_to_optical_flow_model_dir>
```

If you just want to train an optical flow network, adapt `configs/train_flow.yml`, and run:

```
python train_flow.py
```

Note that we use [MLflow](https://mlflow.org/) to keep track of all the experiments. 

## Citations

If you use this library in an academic context, please cite the following:

```
@article{paredes2020back,
  title={Back to Event Basics: Self-Supervised Learning of Image Reconstruction for Event Cameras via Photometric Constancy},
  author={Paredes-Vall{\'e}s, Federico and de Croon, Guido C. H. E.},
  journal={arXiv preprint arXiv:2009.08283},
  year={2020}
}
```

## Acknowledgements

This code borrows from the following open source projects, whom we would like to thank:

- [E2VID](https://github.com/uzh-rpg/rpg_e2vid)
- [Event Contrast Maximization Library](https://github.com/TimoStoff/events_contrast_maximization)