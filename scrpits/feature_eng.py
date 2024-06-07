import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Overview of the dataset
def data_overview(data):
    num_rows, num_cols = data.shape
    data_types = data.dtypes
    print(f"Number of rows: {num_rows}\nNumber of columns: {num_cols}\n")
    print("Data types:")
    print(data_types)
    print("\n")

# Summary Statistics
def summary_statistics(data):
    print("Summary Statistics:")
    print(data.describe())
    print("\n")

# Distribution of Numerical Features
def plot_numerical_distribution(data):
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numerical Features: {numerical_features}")
    for feature in numerical_features:
        plt.figure()
        sns.histplot(data[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
        plt.show()

# Distribution of Categorical Features
def plot_categorical_distribution(data):
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical Features: {categorical_features}")
    for feature in categorical_features:
        plt.figure()
        sns.countplot(y=data[feature], order=data[feature].value_counts().index)
        plt.title(f'Distribution of {feature}')
        plt.show()

# Correlation Analysis
def correlation_analysis(data):
    correlation_matrix = data.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# Identifying Missing Values
def identify_missing_values(data):
    missing_values = data.isnull().sum()
    print("Missing Values:")
    print(missing_values[missing_values > 0])
    print("\n")

# Handling Missing Values: Imputation and Removal
def handle_missing_values(data, strategy='mean'):
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy=strategy)
    data[numerical_features] = imputer.fit_transform(data[numerical_features])
    return data

# Outlier Detection
def detect_outliers(data):
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numerical Features: {numerical_features}")
    for feature in numerical_features:
        plt.figure()
        sns.boxplot(data[feature])
        plt.title(f'Box plot of {feature}')
        plt.show()

# Feature Engineering: Create Aggregate Features
def create_aggregate_features(data):
    data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce')
    aggregate_features = data.groupby('AccountId').agg(
        TotalTransactionAmount=('Amount', 'sum'),
        AverageTransactionAmount=('Amount', 'mean'),
        TransactionCount=('Amount', 'count'),
        StdDevTransactionAmount=('Amount', 'std')
    ).reset_index()
    return aggregate_features

# Feature Extraction: Extract Transaction Hour, Day, Month, Year
def extract_time_features(data):
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    data['TransactionHour'] = data['TransactionStartTime'].dt.hour
    data['TransactionDay'] = data['TransactionStartTime'].dt.day
    data['TransactionMonth'] = data['TransactionStartTime'].dt.month
    data['TransactionYear'] = data['TransactionStartTime'].dt.year
    return data

# Encode Categorical Variables: One-Hot Encoding
def one_hot_encode(data, categorical_features):
    data = pd.get_dummies(data, columns=categorical_features)
    return data

# Encode Categorical Variables: Label Encoding
def label_encode(data, categorical_features):
    le = LabelEncoder()
    for feature in categorical_features:
        data[feature] = le.fit_transform(data[feature].astype(str))
    return data

# Normalize Numerical Features
def normalize_features(data):
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data

# Standardize Numerical Features
def standardize_features(data):
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data

# Main function
def main(file_path):
    data = load_data(file_path)
    data_overview(data)
    #summary_statistics(data)
    #plot_numerical_distribution(data)
    #plot_categorical_distribution(data)
    #correlation_analysis(data)
    identify_missing_values(data)
    
    # Handle Missing Values
    data = handle_missing_values(data, strategy='mean')
    
    #detect_outliers(data)
    
    # Feature Engineering
    aggregate_features = create_aggregate_features(data)
    print("Aggregate Features:")
    print(aggregate_features.head())
    
    # Feature Extraction
    data = extract_time_features(data)
    print("Data with Time Features:")
    print(data[['TransactionStartTime', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']].head())
    
    # Categorical Encoding
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    data_one_hot_encoded = one_hot_encode(data, categorical_features)
    data_label_encoded = label_encode(data, categorical_features)
    
    print("One-Hot Encoded Data:")
    print(data_one_hot_encoded.head())
    
    print("Label Encoded Data:")
    print(data_label_encoded.head())
    
    # Normalize Features
    normalized_data = normalize_features(data.copy())
    print("Normalized Data:")
    print(normalized_data.head())
    
    # Standardize Features
    standardized_data = standardize_features(data.copy())
    print("Standardized Data:")
    print(standardized_data.head())

# Run the main function
if __name__ == "__main__":
    file_path = '../data/data.csv'
    main(file_path)
