import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_csv("processed_data.csv")

# --------------------------
# Feature Engineering
# --------------------------
# Convert to datetime (using existing DaysUntilExpiry)
df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])

# --------------------------
# Prepare Features
# --------------------------
X = df.drop(['ProductID', 'ProductName', 'ExpiryDate', 'HistoricalSales'], axis=1)
y = df['HistoricalSales']

# Convert boolean columns to integers
bool_columns = X.select_dtypes(include='bool').columns
X[bool_columns] = X[bool_columns].astype(int)

# --------------------------
# Train-Validation-Test Split
# --------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5, 
    random_state=42
)

# --------------------------
# Hyperparameter Tuning
# --------------------------
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

model = XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                         cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_

# --------------------------
# Training with Early Stopping
# --------------------------
final_model = XGBRegressor(
    **best_params,
    objective='reg:squarederror',
    random_state=42
)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=True
)

# --------------------------
# Evaluation
# --------------------------
train_pred = final_model.predict(X_train)
test_pred = final_model.predict(X_test)

print(f"\nTrain MAE: {mean_absolute_error(y_train, train_pred):.2f}")
print(f"Test MAE: {mean_absolute_error(y_test, test_pred):.2f}")

# --------------------------
# Visualization
# --------------------------
# Feature Importance
feature_names = X.columns
importances = final_model.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Importance Score')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# --------------------------
# Save Model
# --------------------------
joblib.dump(final_model, "improved_demand_forecast_model.pkl")

print("Model training completed successfully!")