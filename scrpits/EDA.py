# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)
# Function to display an overview of the dataset
def data_columnROW(data):
    num_rows, num_cols = data.shape
    data_types = data.dtypes

    print(f"Number of rows: {num_rows}\nNumber of columns: {num_cols}\n")
    print("Data types:")
    print(data_types)
    print("\n")

    

# 1. Overview of the Data
def data_overview(data):
    print("Dataset Overview:")
    print(data.info())
    print("\n")

# 2. Summary Statistics
def summary_statistics(data):
    print("Summary Statistics:")
    print(data.describe())
    print("\n")

# 3. Distribution of Numerical Features
def plot_numerical_distribution(data):
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numerical Features: {numerical_features}")

    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(numerical_features):
        plt.subplot(5, 4, i + 1)
        sns.histplot(data[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

# 4. Distribution of Categorical Features
def plot_categorical_distribution(data):
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical Features: {categorical_features}")

    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(categorical_features):
        plt.subplot(5, 4, i + 1)
        sns.countplot(y=data[feature], order=data[feature].value_counts().index)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

# 5. Correlation Analysis
def correlation_analysis(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# 6. Identifying Missing Values
def identify_missing_values(data):
    missing_values = data.isnull().sum()
    print("Missing Values:")
    print(missing_values[missing_values > 0])
    print("\n")

# 7. Outlier Detection
def detect_outliers(data):
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(numerical_features):
        plt.subplot(5, 4, i + 1)
        sns.boxplot(data[feature])
        plt.title(f'Box plot of {feature}')
    plt.tight_layout()
    plt.show()

def main(file_path):
    data = load_data(file_path)
    data_columnROW(data)
    data_overview(data)
    summary_statistics(data)
    plot_numerical_distribution(data)
    plot_categorical_distribution(data)
    correlation_analysis(data)
    identify_missing_values(data)
    detect_outliers(data)

if __name__ == "__main__":
    file_path = '../data/data.csv'
    main(file_path)

