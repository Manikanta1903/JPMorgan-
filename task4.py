import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Select FICO and default columns, drop missing values, sort by FICO score
fico = df[['fico_score', 'default']].dropna()
fico = fico.sort_values('fico_score').reset_index(drop=True)

# Function to create buckets using quantiles
def create_fico_buckets(df, n_buckets=10):
    df['bucket'], bins = pd.qcut(df['fico_score'], q=n_buckets, retbins=True, labels=False, duplicates='drop')
    return df, bins

# Function to assign inverse ratings (lower score → higher rating)
def assign_ratings(df):
    max_bucket = df['bucket'].max()
    df['rating'] = max_bucket - df['bucket']
    return df

# Function to apply bucket rating for a new FICO score
def fico_to_rating(fico_score, bins):
    for i in range(len(bins) - 1):
        if fico_score < bins[i + 1]:
            return len(bins) - 2 - i
    return 0
# Step 1: Create buckets and get bin boundaries
fico_bucketed, bins = create_fico_buckets(fico, n_buckets=10)

# Step 2: Assign ratings
fico_rated = assign_ratings(fico_bucketed)

# Step 3: Print the first few rows
print(fico_rated.head())
