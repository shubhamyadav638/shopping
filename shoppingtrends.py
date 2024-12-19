# -*- coding: utf-8 -*-
"""ShoppingTrends.py"""

import pandas as pd
import datetime
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics
import joblib

# File path (ensure this file is in the same directory as the script)
file_path = 'shopping_trends.csv'

# Load dataset
df = pd.read_csv(file_path)
data = pd.DataFrame(df)

# Data preprocessing
data.drop(['Color', 'Shipping Type', 'Subscription Status', 'Promo Code Used', 
           'Item Purchased', 'Discount Applied', 'Location', 
           'Preferred Payment Method'], axis=1, inplace=True)

# Map categorical columns to numerical values
data['Payment Method'] = data['Payment Method'].map({'Cash': 0, 'Credit Card': 1, 'PayPal': 2, 
                                                     'Bank Transfer': 3, 'Venmo': 4, 'Debit Card': 5})
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Frequency of Purchases'] = data['Frequency of Purchases'].map({'Fortnightly': 0, 'Weekly': 1, 
                                                                     'Annually': 2, 'Quarterly': 3, 
                                                                     'Bi-Weekly': 4, 'Monthly': 5, 
                                                                     'Every 3 Months': 6})
data['Size'] = data['Size'].map({'L': 0, 'S': 1, 'M': 2, 'XL': 3})
data['Category'] = data['Category'].map({'Clothing': 0, 'Footwear': 1, 'Outerwear': 2, 'Accessories': 3})
data['Season'] = data['Season'].map({'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3})

# Outlier removal (optional: visualize and remove extreme values if needed)
sns.boxplot(data['Purchase Amount (USD)'], orient='h')

# Split features and target
X = data.drop(['Previous Purchases', 'Purchase Amount (USD)'], axis=1)
Y = data['Purchase Amount (USD)']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Model training
lr = LinearRegression()
lr.fit(X_train, Y_train)

rf = RandomForestRegressor()
rf.fit(X_train, Y_train)

xg = XGBRegressor()
xg_final = xg.fit(X, Y)

# Save the model
joblib.dump(xg_final, 'Shopping_Trends.pkl')

# Load the model
model = joblib.load('Shopping_Trends.pkl')

# Prediction example
data_new = pd.DataFrame({
    'Customer ID': [1],
    'Age': [55],
    'Gender': [0],
    'Category': [0],
    'Size': [0],
    'Season': [0],
    'Review Rating': [3.1],
    'Payment Method': [1],
    'Frequency of Purchases': [0.0]
}, index=[0])

data_new = data_new[['Customer ID', 'Age', 'Gender', 'Category', 'Size', 
                     'Season', 'Review Rating', 'Payment Method', 
                     'Frequency of Purchases']]

# Predict
prediction = model.predict(data_new)
print(f"Predicted Purchase Amount (USD): {prediction[0]:.2f}")
