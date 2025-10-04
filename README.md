[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)    
# Noise-2-Lines

Noise2Lines is a deep learning project that trains neural networks to convert noisy, colorful images into clean line drawings. The project demonstrates pixel-level image transformation using convolutional neural networks, where the model learns to identify and extract line patterns from complex, noisy input images.

<div align="center">
    <img src="resources/Screenshot 01.png" width="500">
    <img src="resources/Screenshot 03.png" width="510">
</div>

<div align="center">
    <img src="resources/Screenshot 04.png" width="500">
    <img src="resources/Screenshot 05.png" width="505">
</div>

<div align="center">
    <img src="resources/Screenshot 07.png" width="500">
    <img src="resources/Screenshot 08.png" width="505">
</div>

## Prerequisites

Before using the Python Reddit Nuker, ensure that you have the following:

1. Python: The program requires **Python (3.13.5 recommended)** to be installed on your system. You can download Python from the official Python website: [python.org](https://www.python.org).
2. [Anaconda or Miniconda](https://www.anaconda.com/docs/main) (recommended for managing environments).
3. [Git](https://git-scm.com/) (for cloning the repository).
4. [CUDA](https://developer.nvidia.com/cuda-downloads) (recommended for running AI model on yor GPU, without this it will run on your CPU).

## Info

If you just want the finished model, you can find it in the root directory of this project as `noisetolines.pth`. 

**Note:** this model is the exact same as `model_7.pth` in the *models_out* directory,

## Setup

To set up the Python Reddit Nuker program, follow these steps:

1. **Clone the repository using git:**

   ```bash
   git clone https://github.com/Maddox-RVS/Noise-2-Line
   cd Noise-2-Line
   ```

2. **Create and activate a conda environment (recommended):**

   ```bash
   conda create -n noise2line python=3.13.5
   conda activate noise2line
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generating a dataset

If you intend to train your own models using this repository, you first need to generate a dataset for the models to learn from. In order to do this, run `create_dataset.py`.

```bash
python create_dataset.py
```

The dataset you generate is variable and can be configured using the two contant variables at the top of the python script.
By default they are:

```python
DATASET_SIZE: int = 1000
IMAGE_SIZE: int = 64
```

`DATASET_SIZE` describes the number of image pairs to generate in your dataset. Each pair consists of a noisy input image and its corresponding clean line target image.

`IMAGE_SIZE` describes the dimensions of each generated image in the dataset. Images are square, so a value of 64 creates 64x64 pixel images.

### Training a model

If [CUDA](https://developer.nvidia.com/cuda-downloads) is installed on your machine, the training process will use your GPU, otherwise it will use your CPU.
In order to start the training process, run `train_model.py`.

```bash
python train_model.py
```

The model training parameters are variable and can be configured using the three constant variables at the top of the python script.
By default they are:

```python
MODELS_SAVE_PATH: pathlib.Path = pathlib.Path("models_out")
EPOCHS: int = 350
LEARNING_RATE: float = 0.00005
```

`MODELS_SAVE_PATH` describes the directory where trained model weights will be saved. If the specified directory doesn't exist, it will be automatically created after the training process.

`EPOCHS` describes the number of complete passes through the entire training dataset that the model will make during the training process. Each epoch allows the model to see and learn from every training example once. More epochs generally lead to better performance but take longer to train and may eventually cause overfitting.

`LEARNING_RATE` describes the rate at which the model learns during training. A lower learning rate means the model makes smaller adjustments to its parameters with each training step, leading to more stable but slower learning. A higher learning rate allows for faster learning but may cause the model to overshoot optimal solutions.

### Testing model performance

To test the performance of any trained models, run `test_model.py`. This will generate new data that the model has never seen before and test the models predictions on this new set of data.
This python script has a single constant variable, which by default is:

```python
MODELS_SAVE_PATH: pathlib.Path = pathlib.Path("models_out")
```

Be sure that the `MODEL_SAVE_PATH` variables are consistent between `train_model.py` and `test_model.py`.