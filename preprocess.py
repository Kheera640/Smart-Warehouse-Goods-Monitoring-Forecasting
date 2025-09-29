import pandas as pd
from datetime import datetime

# Load data
df = pd.read_csv("processed_data.csv")

# Convert ExpiryDate to datetime
df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'])

# Calculate days until expiry
current_date = datetime.now()
df['DaysUntilExpiry'] = (df['ExpiryDate'] - current_date).dt.days

# One-hot encode categories
df = pd.get_dummies(df, columns=['Category'])

# Save processed data
df.to_csv("processed_data.csv", index=False)