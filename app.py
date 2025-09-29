from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load("demand_forecast_model.pkl")

# Define the EXACT feature order expected by the model
FEATURE_ORDER = [
    'QuantityInStock',
    'DaysUntilExpiry',
    'Category_Bakery',
    'Category_Dairy',
    'Category_Meat',
    'Category_Produce'
]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = pd.DataFrame([data], columns=FEATURE_ORDER)
        prediction = model.predict(features)
        
        return jsonify({
            "predicted_sales": float(prediction[0]),
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
