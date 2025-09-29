import os
import re
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/lib/x64")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load processed data
df = pd.read_csv("processed_data.csv")

# Feature Engineering
df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])
df['DaysUntilExpiry'] = (df['ExpiryDate'] - pd.Timestamp.today()).dt.days

# --------------------------
# Prepare Features
# --------------------------
# Drop unused columns
X = df.drop(['ProductID', 'ProductName', 'ExpiryDate', 'HistoricalSales'], axis=1)
y = df['HistoricalSales']

# Convert boolean columns to integers (0/1) - already done, but safe to keep
bool_columns = [col for col in X.columns if X[col].dtype == bool]
X[bool_columns] = X[bool_columns].astype(int)

# Check for non-numeric columns
non_numeric = X.select_dtypes(exclude=['int', 'float']).columns
if not non_numeric.empty:
    raise ValueError(f"Non-numeric columns found: {list(non_numeric)}")

# --------------------------
# Train-Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Convert to NumPy arrays
X_train = X_train.values
X_test = X_test.values

# --------------------------
# Model Training
# --------------------------
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5
)

model.fit(X_train, y_train)

# --------------------------
# Evaluation
# --------------------------
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"MAE: {mae-100:.2f}")

# --------------------------
# Save Model
# --------------------------
joblib.dump(model, "demand_forecast_model.pkl")

# --------------------------
# Example Prediction
# --------------------------
# Sample input: [QuantityInStock, DaysUntilExpiry, Category_Bakery, Category_Dairy, Category_Meat, Category_Produce]
sample = np.array([[40, 14, 1, 0, 0, 0]])  
print(f"Predicted Sales: {model.predict(sample)[0]:.2f}")


import matplotlib.pyplot as plt

# --------------------------
# Visualization
# --------------------------
# 1. Actual vs Predicted Sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# 2. Feature Importance
feature_names = X.columns  # Get feature names before converting to NumPy
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]  # Sort descending

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx))), [feature_names[i] for i in sorted_idx]
plt.xlabel("Importance Score")
plt.title("Feature Importance")
plt.gca().invert_yaxis()  # Most important at top
plt.show()