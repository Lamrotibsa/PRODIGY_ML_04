# Gesture Recognition Using CNN



## Project Overview
The goal of this project is to build a robust gesture recognition system using a CNN. The model is trained on grayscale images of different hand gestures and is capable of accurately predicting the gesture shown in a new image.

## Dataset
The dataset used for this project is the LeapGestRecog dataset. It contains grayscale images of 10 different hand gestures. Each gesture is performed by multiple subjects, and the images are organized into folders by subject and gesture.

## Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/Lamrotibsa/PRODIGY_ML_04.git
    cd gesture-recognition-cnn
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Preprocess and Load Data**
    - Run the script to preprocess the data and save it as numpy arrays.

2. **Train the Model**
    - Train the model using the preprocessed data and save the trained model.

3. **Evaluate the Model**
    - Evaluate the model's performance on the test set, including generating a confusion matrix and classification report.

4. **Make Predictions**
    - Use the trained model to make predictions on new images.

## Model Architecture
The model uses a series of convolutional and max-pooling layers followed by dense layers and dropout for regularization. The architecture is designed to effectively capture features from grayscale images of hand gestures.

## Training the Model
The model is trained using data augmentation to prevent overfitting. The `ImageDataGenerator` is used to apply random transformations to the images during training.

## Evaluation
The model's performance is evaluated on a separate test set. The evaluation metrics include accuracy, loss, confusion matrix, and classification report.

## Predictions
You can use the trained model to make predictions on new images. Simply load an image, preprocess it, and use the model to predict the gesture.

## Results
The model achieves high accuracy on the test set, with detailed performance metrics provided in the confusion matrix and classification report.


