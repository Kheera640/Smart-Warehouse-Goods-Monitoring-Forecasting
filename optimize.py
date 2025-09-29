import pandas as pd
import joblib

# Load model and data
model = joblib.load("demand_forecast_model.pkl")
df = pd.read_csv("processed_data.csv")

# Predict sales
X = df.drop(['ProductID', 'ProductName', 'ExpiryDate', 'HistoricalSales'], axis=1)
df['PredictedSales'] = model.predict(X)

# Calculate optimal order quantity
df['OptimalOrder'] = df.apply(lambda row: max(0, row['PredictedSales'] - row['QuantityInStock'] + 10), axis=1)  # 10 = safety stock

# Generate expiry alerts (e.g., alert if < 7 days left)
df['ExpiryAlert'] = df['DaysUntilExpiry'].apply(lambda x: 'YES' if x <= 7 else 'NO')

# Save results
df.to_csv("inventory_recommendations.csv", index=False)