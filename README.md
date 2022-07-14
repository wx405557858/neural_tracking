# Marker Tracking with Neural Networks

This project provides robust marker tracking for camera-based tactile sensors with Sim2Real neural networks.

## Getting Started

<img src="https://github.com/wx405557858/neural_tracking_data/blob/media/imgs/output_example.gif" align="right" width=384>

**System**: Ubuntu, OSX

**Install packages**

```
pip install tensorflow-gpu opencv-python
```

**Download models**

```
bash ./models/download_model.sh
```

**Download test GelSight video**

```
bash ./data/download_video.sh
```

**Marker tracking for GelSight video**

```
python example_tracking_video.py
```

**Note**: The model is implemented initially with Keras/TensorFlow to be compatible with [Coral](https://coral.ai/products/accelerator), using TPU as USB Accelerator for Raspberry Pi on-device computation. Please feel free to switch the model to other frameworks, like PyTorch, for your purpose.



## Examples
 
* `python example_tracking_sim.py`

The interactive tracking demo. The mouse distorts (translates and rotates) the marker flow. The yellow arrow shows the marker tracking predictions from the neural network. The model can robustly track markers, even with extreme and complex interactions. The model is trained with 10x14 markers.

<img src="https://github.com/wx405557858/neural_tracking_data/blob/media/imgs/output_sim_example.gif" width=384>

The model is also robust to marker sizes and background disturbances, due to added domain randomization during training.

<img src="https://github.com/wx405557858/neural_tracking_data/blob/media/imgs/output_sim_example_disturb.gif" width=384>

* `python example_tracking_sim_generic.py`

The generic model is trained on variable grid patterns so that it can be invariant to different numbers of markers. The output is the flow with the same size of the input. 

<img src="https://github.com/wx405557858/neural_tracking_data/blob/media/imgs/output_sim_generic_example_disturb.gif" width=384>


**Note**: We suggest trying the generic model for preliminary experiments, and training your fixed model for best performance. The generic one can work on more cases directly, and the fixed one is more accurate for a certain marker pattern.

* `python example_tracking_video.py`

The model can be transferred to real sensor data robustly, with large forces, multiple contacts, and wrinkles.

<img src="https://github.com/wx405557858/neural_tracking_data/blob/media/imgs/output_example.gif" width=384>

## Train

You can train your own model to optimize the performance for a specific marker pattern. Here are two training example that is trained on 10x14 markers (`train.py`), and variable marker patterns (`train_generic.py`). Please feel free to customize them for your purposes.

### `python train.py`

It takes data pairs `(X, Y)` from `generate_img()` in `generate_data.py`. 

* **Input**: `X` is the synthesized image. It has random backgrounds, and 10x14 markers. The marker positions are randomly distorted (translated and rotated) with Gaussian-based smoothing. It imitates smooth elastomer distortions with simple operations. The markers are then rendered as regions darker than surrounding pixels, given the distorted positions.
* **Output**: `Y` has a dimension of 10x14x2, which represents the ground-truth horizontal and vertical displacement in pixel for each marker.
* **Model**: The model consists multiple Convolutional layers and Pooling layers, defined as `build_model_small()` in `train.py`.

### `python train_generic.py`

It takes the data pairs `(X, Y)` from `generate_img()` in `generate_data_generic.py`.

* **Input**: `X` is generated similarly to the previous one. The difference is that the number of rows and columns is randomly selected from [4, 15].
* **Output**: `Y` is the flow with the same width and height as the input image. Each point represents the horizontal and vertical displacement for each pixel. We use multiple resolutions of Y to accelerate training.
* **Model**: The model is in the Encoder-Decoder style, which consists of Convolutional, Pooling, and Upsampling layers to generate output with the same dimensions as the input. It is defined as `build_model_ae` in train.py

**Note**: Please customize the `generate_img()` function to fit the marker patterns to your sensor.

## Citation
If you use this code for your research, please cite our paper: [Gelsight Wedge: Measuring High-Resolution 3D Contact Geometry with a Compact Robot Finger](https://arxiv.org/pdf/2106.08851.pdf):

```
@inproceedings{wang2021gelsight,
  title={Gelsight wedge: Measuring high-resolution 3d contact geometry with a compact robot finger},
  author={Wang, Shaoxiong and She, Yu and Romero, Branden and Adelson, Edward},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={6468--6475},
  year={2021},
  organization={IEEE}
}
```
