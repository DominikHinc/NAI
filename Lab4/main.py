"""

Sylwia Juda (s25373) & Dominik Hinc (s22436)

Problem:
- Data Classification with Decision Tree and SVM

Program Overview:
  This program is designed to classify data using Decision Tree and SVM classifiers. The program performs the following steps:

  1. Load data from CSV files:
     - The first dataset is loaded from "auto-insurance.csv".
     - The second dataset is loaded from "insurance_claims.csv".

  2. Display scatter plots of the data to visualize the relationship between the features and the target variables.

  3. Prepare the data for training:
     - Extract the feature column (X) and convert the target column (Y) into binary classes based on a specified threshold.

  4. Split the data into training and testing sets.

  5. Train and evaluate classifiers:
     - Train a Decision Tree classifier on the training data and evaluate its performance on the test data.
     - Train a SVM classifier on the training data and evaluate its performance on the test data.

  6. Print evaluation metrics:
     - Accuracy score
     - Classification report
     - Confusion matrix

By default the program is performing the following classification tasks:
  - The first dataset ("auto-insurance.csv") contains data on the number of claims (X) and the cost of claims (Y). The analysis aims to classify whether the cost of claims is high (Y > 100) or low (Y <= 100).
  - The second dataset ("insurance_claims.csv") contains data on the age of claimants (age) and the total claim amount (total_claim_amount). The analysis aims to classify whether the total claim amount is high (total_claim_amount > 60000) or low (total_claim_amount <= 60000).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



def load_data_from_file(fileName):
    """
    Load data from a file and return as a DataFrame.

    :param fileName: The name of the file to load data from.

    :return: DataFrame containing the data.
    """

    data = pd.read_csv(fileName)
    df = pd.DataFrame(data)
    return df

def display_scatter_plot(df, column1, column2, title1, title2):
    """
    Display the data in a scatter plot.
    """
    plt.scatter(df[column1], df[column2], color='blue', label='Dane')
    plt.xlabel(title1)
    plt.ylabel(title2)
    plt.title("Rozrzut danych")
    plt.show()

def prepare_data(df, column1, column2, threshold):
    """
    Prepare the data for training. Extract feature column and convert target column to binary.

    :param df: The DataFrame containing the data.
    :param column1: The name of the column to use as X.
    :param column2: The name of the column to use as y.
    :param threshold: The threshold value to use for classification.
    
    :return: The X and y values.
    """
    data_column_1 = df[[column1]] 
    data_column_2 = np.where(df[column2] > threshold, 1, 0)
    return data_column_1, data_column_2

def train_decision_tree(column1_train, column2_train):
    """
    Trains a Decision Tree classifier.

    :param column1_train: The training data for column 1.
    :param column2_train: The training data for column 2.

    :return: The trained classifier.
    """
    classifier = DecisionTreeClassifier()
    classifier.fit(column1_train, column2_train)
    return classifier

def train_svm_classifier(column1_train, column2_train):
    """
    Trains a SVM classifier.

    :param column1_train: The training data for column 1.
    :param column2_train: The training data for column 2.

    :return: The trained classifier.
    """
    classifier = SVC()
    classifier.fit(column1_train, column2_train)
    return classifier

def evaluate_classifier(classifier, column1_test, column2_test):
    """
    Evaluate the classifier using test data.

    :param classifier: The trained classifier.
    :param column1_test: The test data for column 1.
    :param column2_test: The test data for column 2.

    :return: The evaluation metrics.
    """
    y_pred = classifier.predict(column1_test)
    print("Accuracy Score:", accuracy_score(column2_test, y_pred))
    print("Classification Report:\n", classification_report(column2_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(column2_test, y_pred))

def main():
    """
     Main function to run the program.
    """

    # FIRST DATA SET

    file_name = "auto-insurance.csv"

    column1 = 'X'
    column2 = 'Y'
    df = load_data_from_file(file_name)

    display_scatter_plot(df, column1, column2,  "Liczba roszczeń (X)", "Koszt roszczeń (Y)")
    x_data, y_data = prepare_data(df, column1, column2, 100)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    print("Drzewo decyzyjne dla danych " + file_name + ", kolumny " + column1 + " i " + column2 + ":")
    decision_tree_classifier = train_decision_tree(x_train, y_train)
    evaluate_classifier(decision_tree_classifier, x_test, y_test)

    
    print("\nSVM dla danych " + file_name + ", kolumny " + column1 + " i " + column2 + ":")
    svm_classifier = train_svm_classifier(x_train, y_train)
    evaluate_classifier(svm_classifier, x_test, y_test)

    # SECOND DATA SET

    file_name2 = "insurance_claims.csv"
    column1 = 'age'
    column2 = 'total_claim_amount'

    df2 = load_data_from_file(file_name2)

    display_scatter_plot(df2, column1, column2, "Wiek", "Koszt roszczeń")
    x_data, y_data = prepare_data(df2, column1, column2, 60000)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    print("\nDrzewo decyzyjne dla danych " + file_name2 + ", kolumny " + column1 + " i " + column2 + ":")
    decision_tree_classifier = train_decision_tree(x_train, y_train)
    evaluate_classifier(decision_tree_classifier, x_test, y_test)

    print("\nSVM dla danych " + file_name2 + ", kolumny " + column1 + " i " + column2 + ":")
    svm_classifier = train_svm_classifier(x_train, y_train)
    evaluate_classifier(svm_classifier, x_test, y_test)



main()
