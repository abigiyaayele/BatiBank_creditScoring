#(new_data_EDA.ipynb)
the new data that is modifed the given dataset that has column names plus number
def extract_numeric_parts(data):
    data['TransactionId'] = data['TransactionId'].str.extract('(\d+)') # Extract only numeric part
    data['BatchId'] = data['BatchId'].str.extract('(\d+)') # Extract only numeric part
    data['AccountId'] = data['AccountId'].str.extract('(\d+)') # Extract only numeric part
    data['SubscriptionId'] = data['SubscriptionId'].str.extract('(\d+)') # Extract only numeric part
    data['CustomerId'] = data['CustomerId'].str.extract('(\d+)') # Extract only numeric part
    data['ProviderId'] = data['ProviderId'].str.extract('(\d+)') # Extract only numeric part
    data['ProductId'] = data['ProductId'].str.extract('(\d+)') # Extract only numeric part
    data['ChannelId'] = data['ChannelId'].str.extract('(\d+)') # Extract only numeric part
    then save data as data2.csv
     
      this is the main data  that this task 1 EDA is performed
# Explanation of the Functions
### load_data(file_path):
 Loads the dataset from the specified file path.
### data_overview(data): 
Prints an overview of the dataset, including the data types and non-null counts.
### summary_statistics(data):
 Prints the summary statistics for the numerical features in the dataset.
### plot_numerical_distribution(data):
 Plots the distribution of numerical features using histograms and KDE plots.
### plot_categorical_distribution(data):
 Plots the distribution of categorical features using count plots.
### correlation_analysis(data):
 Displays the correlation matrix of numerical features using a heatmap.
### identify_missing_values(data):
 Identifies and prints the number of missing values in each column.
### detect_outliers(data): Detects and visualizes outliers in numerical features using box plots.