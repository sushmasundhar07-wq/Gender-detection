# Gender Detection Using Deep Learning

A Python-based gender classification project that uses **TensorFlow, Keras, and MobileNetV2** to classify face images into the dataset's **Female** or **Male** labels. The project supports testing with a webcam.

## Features

* Image classification using a pretrained MobileNetV2 model
* Transfer learning with TensorFlow and Keras
* Training, validation, and testing using separate datasets
* Real-time webcam prediction using OpenCV

## Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV2
* OpenCV
* NumPy

## Project Structure

```text
Gender-detection/
├── dataset/
│   ├── train/
│   │   ├── Female/
│   │   └── Male/
│   └── test/
│       ├── Female/
│       └── Male/
├── train.py
├── predict.py
├── gender_model.keras
├── .gitignore
└── README.md
```

> The dataset, virtual environment, and trained model are excluded from GitHub by `.gitignore`. The model file is generated after training.

## Requirements

* Python 3.10 or another Python version compatible with your installed TensorFlow release
* A webcam for live testing
* Git (optional, for version control)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sushmasundhar07-wq/Gender-detection.git
cd Gender-detection
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install tensorflow opencv-python numpy
```

## Dataset Setup

Place your images in the following folder structure:

```text
dataset/
├── train/
│   ├── Female/
│   └── Male/
└── test/
    ├── Female/
    └── Male/
```

The folder names are used as class labels. Keep the labels consistent between the training and testing folders.

## Train the Model

Run:

```bash
python train.py
```

The training script loads the images, trains the model using MobileNetV2, evaluates it on the test dataset, and saves the trained model as:

```text
gender_model.keras
```

## Test Using Your Webcam

After training, run:

```bash
python predict.py
```

The webcam window displays the model's predicted class and its score.

Press **Q** to close the webcam window.

## Important Notes

* The model predicts the labels represented in its training dataset; it does not determine a person's gender identity.
* Predictions may be incorrect due to dataset quality, lighting, camera angle, image quality, or differences between training images and webcam images.
* The displayed confidence score is the model's output-derived score, not a guarantee that the prediction is correct.
* The dataset and trained model are not included in this repository. You must provide the dataset and train the model before running webcam prediction.

## Future Improvements

* Improve dataset quality and diversity
* Add face detection before classification
* Evaluate performance on varied lighting and camera conditions
* Add a graphical user interface

## Author

@sushmasundhar07

