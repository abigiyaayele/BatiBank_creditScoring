
import pandas as pd
import numpy as np
from xverse.transformer import WOE

# Define functions for WOE calculation

def calculate_woe_continuous(data, feature, target):
    data['bins'], bins = pd.qcut(data[feature], 10, retbins=True, duplicates='drop')
    
    grouped = data.groupby(['bins', target], observed=True).size().unstack().fillna(0)
    
    # Check if grouped DataFrame is empty
    if grouped.empty:
        default_values = pd.DataFrame({
            'event_perc': [0.5],  # Default value for event percentage
            'non_event_perc': [0.5],  # Default value for non-event percentage
            'woe': [0.0]  # Default value for WOE
        })
        return default_values['woe']
    
    # Calculate percentage of events and non-events in each bin
    grouped['event_perc'] = grouped[1] / (grouped[0] + grouped[1] + 0.001)  # Add 0.001 to avoid division by zero
    grouped['non_event_perc'] = grouped[0] / (grouped[0] + grouped[1] + 0.001)
    
    # Calculate WOE
    grouped['woe'] = np.log(grouped['non_event_perc'] / grouped['event_perc'] + 0.001)  # Add 0.001 to avoid log(0)
    
    return grouped['woe']

def calculate_woe_categorical(df, feature, target):
    grouped = data.groupby([feature, target], observed=True).size().unstack().fillna(0)
    
    # Calculate percentage of events and non-events for each category
    grouped['event_perc'] = grouped[1] / (grouped[0] + grouped[1] + 0.001)  # Add 0.001 to avoid division by zero
    grouped['non_event_perc'] = grouped[0] / (grouped[0] + grouped[1] + 0.001)
    
    # Calculate WOE
    grouped['woe'] = np.log(grouped['non_event_perc'] / grouped['event_perc'] + 0.001)  # Add 0.001 to avoid log(0)
    
    return grouped['woe']

# Function to preprocess data using WOE

def preprocess_data(data):
    woe_data = data.copy()
    
    # Check if 'FraudResult' column exists
    if 'FraudResult' not in woe_data.columns:
        raise ValueError("Column 'FraudResult' not found in the data.")
    
    # Convert 'TransactionStartTime' to datetime
    woe_data['TransactionStartTime'] = pd.to_datetime(woe_data['TransactionStartTime'])
    
    # Calculate WOE for each feature
    for feature in data.columns:
        if feature == 'FraudResult':
            continue  # Skip the target variable
        if data[feature].dtype == 'object':
            woe_data[feature + '_woe'] = calculate_woe_categorical(data, feature, 'FraudResult')
        else:
            woe_data[feature + '_woe'] = calculate_woe_continuous(data, feature, 'FraudResult')
    
    return woe_data

# Function to calculate RFM

def calculate_rfm(df, reference_date):
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (reference_date - x.max()).days,  # Recency
        'CustomerId': 'count',                                             # Frequency
        'Amount': ['sum', 'mean']                                          # Monetary and Spend
    }).reset_index()
    rfm.columns = ['CustomerId', 'recency', 'frequency', 'monetary', 'spend']
    rfm.fillna(0, inplace=True)
    return rfm

# Function to calculate RFM score

def calculate_rfms_score(rfm):
    rfm['rfms_score'] = (0.25 * rfm['recency']) + (0.25 * rfm['frequency']) + (0.25 * rfm['monetary']) + (0.25 * rfm['spend'])
    return rfm

# Function to classify users based on RFM score and threshold

def classify_users(rfm, threshold):
    rfm['user_class'] = rfm['rfms_score'].apply(lambda x: 'good' if x >= threshold else 'bad')
    return rfm


if name == "main":
    # Read data
    data = pd.read_csv('creditScoring/data/raw/data.csv')
    
    # Preprocess data using WOE
    data_preprocessed = preprocess_data(data)
    
    # Set reference date for RFM calculation (for example, today's date)
    reference_date = pd.to_datetime('2018-11-15T02:18:49Z')
    
    # Calculate RFM
    rfm_data = calculate_rfm(data_preprocessed, reference_date)
    
    # Calculate RFM score
    rfm_data_scored = calculate_rfms_score(rfm_data)
    
    # Classify users
    threshold = 500  # Define your threshold
    classified_users = classify_users(rfm_data_scored, threshold)
    
    print(classified_users.head())  # Print the first few rows

