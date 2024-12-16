""""
Sylwia Juda (s25373) & Dominik Hinc (s22436)

Problem:
- Image Prediction with Neural Networks

Program Overview:
    This program is designed to classify images using a convolutional neural network. The program performs the following steps:
    1. Load the MNIST dataset and filter it for specific classes.
    2. Build and compile a CNN model for image classification.
    3. Train the model on the filtered dataset.
    4. Save the trained model to a file.
    5. Load and preprocess custom images for prediction.
    6. Use the model to predict the classes of the custom images.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import numpy as np
from keras.datasets import mnist

CLASSES = list(range(10))
IMG_SIZE = 28

def load_mnist():
    """
    Load the MNIST dataset.

    :return: Training and testing datasets.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    y_train = to_categorical(y_train, num_classes=len(CLASSES))
    y_test = to_categorical(y_test, num_classes=len(CLASSES))

    return x_train, y_train, x_test, y_test

def build_model(input_shape, num_classes):
    """
    Build and compile a CNN model for image classification.

    :param input_shape: The shape of the input images.
    :param num_classes: The number of classes to predict.

    :return: The compiled model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image_path):
    """
    Preprocess a single image for prediction.

    :param image_path: The path to the image file.

    :return: The preprocessed image.
    """
    img = load_img(image_path, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=-1)

def main():
    """
    Train the model on MNIST and use it to predict custom images.
    """

    print("Ładowanie danych...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"Zbiór danych załadowany! Trenowanie na {len(CLASSES)} klasach: {CLASSES}")

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    model = build_model(x_train.shape[1:], len(CLASSES))
    print("Trenowanie modelu...")
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

    model.save('mnist_model.h5')
    print("Model zapisano jako 'mnist_model.h5'.")

    print("Ładowanie i przetwarzanie obrazów...")
    digit_5 = preprocess_image('digit-5.png')
    digit_8 = preprocess_image('digit-8.png')

    print("Predykcja obrazów...")
    digit_5_prediction = model.predict(np.expand_dims(digit_5, axis=0))
    digit_8_prediction = model.predict(np.expand_dims(digit_8, axis=0))

    digit_5_class = np.argmax(digit_5_prediction)
    digit_8_class = np.argmax(digit_8_prediction)

    print(f"digit-5.png został zidentyfikowany jako: {digit_5_class}")
    print(f"digit-8.png został zidentyfikowany jako: {digit_8_class}")

main()
