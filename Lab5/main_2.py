"""
Sylwia Juda (s25373) & Dominik Hinc (s22436)

Problem:
- Image Prediction with Neural Networks

Program Overview:
    This program is designed to classify images using a convolutional neural network. The program performs the following steps:
    1. Load the CIFAR-10 dataset and filter it for specific classes.
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
from sklearn.preprocessing import LabelEncoder
import numpy as np

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
IMG_SIZE = 32 

def load_data(classes):
    """
    Load the CIFAR-10 dataset and filter it for specific classes.

    :param classes: List of classes to filter the dataset for.

    :return: Tuple of (x_train, y_train, x_test, y_test) containing the filtered dataset.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    class_indices = [cifar10_classes.index(cls) for cls in classes]
    
    train_mask = np.isin(y_train, class_indices).flatten()
    test_mask = np.isin(y_test, class_indices).flatten()
    
    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]
    
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    label_encoder = LabelEncoder()
    label_encoder.fit(class_indices)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
    
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
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def main():
    """
    Train the model on CIFAR-10 and use it to predict custom images.
    """

    print("Ładowanie danych...")
    x_train, y_train, x_test, y_test = load_data(CLASSES)
    print(f"Zbiór danych załadowany! Trenowanie nia {len(CLASSES)} klasach: {CLASSES}")

    model = build_model(x_train.shape[1:], len(CLASSES))
    print("Trenowanie modelu...")
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))
    
    model.save('cifar10_animal_model.h5')
    print("Model zapisany poprawnie jako 'cifar10_animal_model.h5'.")

    print("Ładowanie i przetwarzanie obrazów...")
    dog_image = preprocess_image('dog.png')
    cat_image = preprocess_image('cat.png')

    print("Predykcja obrazów...")
    dog_prediction = model.predict(dog_image)
    cat_prediction = model.predict(cat_image)

    dog_class = CLASSES[np.argmax(dog_prediction)]
    cat_class = CLASSES[np.argmax(cat_prediction)]

    print(f"dog.png został zidentyfikowany jako: {dog_class}")
    print(f"cat.png został zidentyfikowany jako: {cat_class}")

main()
