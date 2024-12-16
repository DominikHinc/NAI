""""
Sylwia Juda (s25373) & Dominik Hinc (s22436)

Problem:
- Image Prediction with Neural Networks

Program Overview:
    This program is designed to classify images using a convolutional neural network. The program performs the following steps:
    1. Load the Fashion-MNIST dataset and filter it for specific classes.
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

CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat'] 
IMG_SIZE = 28 

def load_fashion_mnist(classes):
    """
    Load the Fashion-MNIST dataset and filter it for specific classes.

    :param classes: List of class names to filter.

    :return: Filtered training and testing datasets.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    class_indices = [i for i, cls in enumerate(CLASSES)]

    train_mask = np.isin(y_train, class_indices)
    test_mask = np.isin(y_test, class_indices)

    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    label_mapping = {old: new for new, old in enumerate(class_indices)}
    y_train = np.array([label_mapping[label] for label in y_train])
    y_test = np.array([label_mapping[label] for label in y_test])

    y_train = to_categorical(y_train, num_classes=len(classes))
    y_test = to_categorical(y_test, num_classes=len(classes))

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
    Train the model on Fashion-MNIST and use it to predict custom images.
    """

    print("Ładowanie danych...")
    x_train, y_train, x_test, y_test = load_fashion_mnist(CLASSES)
    print(f"Zbiór danych załadowany! Trenowanie nia {len(CLASSES)} klasach: {CLASSES}")

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    model = build_model(x_train.shape[1:], len(CLASSES))
    print("Trenowanie modelu...")
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

    model.save('fashion_mnist_model.h5')
    print("Model zapisano jako 'fashion_mnist_model.h5'.")

    print("Ładowanie i przetwarzanie obrazów...")
    tshirt_image = preprocess_image('tshirt.png')
    dress_image = preprocess_image('dress.png')

    print("Predykcja obrazów...")
    tshirt_prediction = model.predict(np.expand_dims(tshirt_image, axis=0))
    dress_prediction = model.predict(np.expand_dims(dress_image, axis=0))

    tshirt_class = CLASSES[np.argmax(tshirt_prediction)]
    dress_class = CLASSES[np.argmax(dress_prediction)]

    print(f"tshirt.png został zidentyfikowany jako: {tshirt_class}")
    print(f"dress.png został zidentyfikowany jako: {dress_class}")

main()
