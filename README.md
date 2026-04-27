# Chest X-ray Pneumonia Detection

Binary image classification of chest X-rays as **NORMAL** or **PNEUMONIA** using a convolutional neural network built with Keras / TensorFlow. Trained end-to-end in Google Colab.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney on Kaggle. Downloaded inside the notebook via the Kaggle API.

Split sizes (as observed in the notebook):

- Train: 5,216 images
- Validation: 16 images
- Test: 624 images

Images are organized into `NORMAL/` and `PNEUMONIA/` subfolders under each split.

## Tech stack

- Python
- Keras (TensorFlow backend)
- NumPy, Matplotlib, PIL
- Kaggle API (for dataset download)
- Google Colab (training environment)

## Model

Simple Sequential CNN:

```
Conv2D(32, 3x3, ReLU)  -> MaxPool(2x2)
Conv2D(32, 3x3, ReLU)  -> MaxPool(2x2)
Flatten
Dense(128, ReLU)
Dense(1,   Sigmoid)
```

- Input: 64x64 RGB images
- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Metric: `accuracy`

Training uses `ImageDataGenerator` with rescaling (1/255), shear, zoom, and horizontal flip for augmentation. The model is trained for **40 epochs** with `steps_per_epoch=163` and `batch_size=32`.

## Results

After 40 epochs (final epoch, as logged in the notebook):

- Training accuracy: ~96.9%
- Validation accuracy: ~81.3%

Validation accuracy is noisy across epochs because the official Kaggle validation split contains only 16 images.

## How to run

The whole project is contained in `ml_final.ipynb` and was authored against Google Colab.

1. Open `ml_final.ipynb` in Google Colab (or a local Jupyter environment with Keras + TensorFlow installed).
2. Provide a Kaggle API token (`kaggle.json`) when the first cell prompts for an upload.
3. Run the cells top-to-bottom. The notebook will:
   - Install the Kaggle CLI
   - Download and unzip the `paultimothymooney/chest-xray-pneumonia` dataset
   - Build the CNN, train it, and plot accuracy curves

## Files

- `ml_final.ipynb` — full notebook (data download, preprocessing, model, training, evaluation)
- `kaggle.json` — Kaggle API credentials (note: this contains a token and should generally not be committed; rotate it if it has been used)
