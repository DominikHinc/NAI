"""
Sylwia Juda (s25373) & Dominik Hinc (s22436)

Problem:
- Data Classification with Neural Networks

Program Overview:
    This program is designed to classify data using a simple feedforward neural network. The program performs the following steps:

    1. Load data from a CSV file:
        - The dataset is loaded from "auto-insurance.csv".

    2. Prepare the data for training:
        - Extract the feature column (X) and convert the target column (Y) into binary classes based on a specified threshold.

    3. Split the data into training and testing sets.

    4. Build and train a neural network:
        - The neural network uses a simple feedforward architecture with 16 units in the first hidden layer, 8 units in the second hidden layer, ReLU activation functions for hidden layers, and a sigmoid activation function for the output layer.
        - The network uses binary crossentropy loss function, L2 regularization with a factor of 0.01 for all hidden layers, and the Adam optimizer.

    5. Evaluate the neural network:
        - Print the accuracy score, classification report, and confusion matrix.
"""


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2



def load_data_from_file(file_name):
    """
    Load data from a CSV file and return as a DataFrame.

    :param: file_name: The name of the CSV file to load.

    :return: DataFrame containing the data.
    """
    return pd.read_csv(file_name)

def prepare_data(df, feature_column, target_column, threshold):
    """
    Prepare the data by extracting features and binarizing the target column.

    :param df: The DataFrame containing the data.
    :param feature_column: The name of the feature column.
    :param target_column: The name of the target column.
    :param threshold: The threshold value to use for binarization.

    :return: The feature matrix X and the target vector y.
    """
    X = df[[feature_column]].values
    y = np.where(df[target_column] > threshold, 1, 0)
    return X, y

def build_neural_network(input_dim):
    """
    Build a simple feedforward neural network.
    The network uses:
    - 16 units in the first hidden layer
    - 8 units in the second hidden layer
    - ReLU activation functions for hidden layers
    - Sigmoid activation function for the output layer
    - Binary crossentropy loss function
    - L2 regularization with a factor of 0.01 for all hidden layers
    - Adam optimizer

    :param input_dim: The number of input features.

    :return: The compiled neural network model.
    """
    model = Sequential([
        Input(shape=(input_dim,)),  # Explicit input layer
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print metrics.

    :param model: The trained model.
    :param X_test: The test feature matrix.
    :param y_test: The test target vector.
    """
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def main():
    # Load and prepare the first dataset
    file_name = "auto-insurance.csv"
    df = load_data_from_file(file_name)
    
    column1 = 'X'
    column2 = 'Y'
    threshold = 100
    
    X, y = prepare_data(df, column1, column2, threshold)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train the neural network
    model = build_neural_network(input_dim=X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    
    # Evaluate the neural network
    print(f"Ewaluacja Sieci Neuronowej dla {file_name}:")
    evaluate_model(model, X_test, y_test)


main()
