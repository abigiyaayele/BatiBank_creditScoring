import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
# Load your data
data = pd.read_csv('data2.csv')  # Replace 'your_data.csv' with your actual file

# Feature Engineering (based on correlation insights)
# Example: Combine 'Amount' and 'Value' into a single feature
data['TransactionValue'] = data['Amount']  # Assuming 'Amount' is more relevant

# Drop redundant features
data.drop(['Value', 'Risk'], axis=1, inplace=True)  # Assuming 'FraudResult' is sufficient

# Feature Encoding (if needed)
# Encode Categorical Variables: One-Hot Encoding
def one_hot_encode(data, categorical_features):
    data = pd.get_dummies(data, columns=categorical_features)
    return data

categorical_features = data.select_dtypes(include=['object']).columns.tolist()
data_one_hot_encoded = one_hot_encode(data, categorical_features)
print("One-Hot Encoded Data:")
print(data_one_hot_encoded.head())
# Example: One-hot encode categorical features
# Encode Categorical Variables: Label Encoding
def label_encode(data, categorical_features):
    le = LabelEncoder()
    for feature in categorical_features:
        data[feature] = le.fit_transform(data[feature].astype(str))
    return data

data_label_encoded = label_encode(data, categorical_features)
print("Label Encoded Data:")
print(data_label_encoded.head())
# data = pd.get_dummies(data, columns=['ProductCategory', 'Channelld'], drop_first=True)

# Split data into features (X) and target (y)
X = data.drop('FraudResult', axis=1)
y = data['FraudResult']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting Machine': GradientBoostingClassifier()
}

# Hyperparameter Tuning (Example: Grid Search for Logistic Regression)
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']  # Solvers suitable for L1 regularization
}
grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_log_reg = grid_search.best_estimator_

# Train models (using best parameters for Logistic Regression)
for name, model in models.items():
    if name == 'Logistic Regression':
        model = best_log_reg
    model.fit(X_train, y_train)

# Evaluate models
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"{name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC-AUC: {roc_auc}")

# Choose the best model based on evaluation metrics (e.g., Random Forest)
best_model = models['Random Forest']

# Predict probabilities (for credit score assignment)
probabilities = best_model.predict_proba(X_test)[:, 1]

# Credit Score Mapping
def risk_to_credit_score(prob):
    return 800 - prob * 700  # Adjust based on your business logic

credit_scores = np.array([risk_to_credit_score(p) for p in probabilities])
print(credit_scores)

# Loan Amount and Duration Prediction (Placeholder)
def predict_loan_amount_and_duration(prob):
    if prob < 0.2:
        return 10000, 36
    elif prob < 0.5:
        return 5000, 24
    else:
        return 1000, 12

loan_predictions = np.array([predict_loan_amount_and_duration(p) for p in probabilities])
print(loan_predictions)