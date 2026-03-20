# app.py - COMPLETE Flask Backend
from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS
import numpy as np
from utils import load_model, validate_and_prepare_input

app = Flask(__name__)
CORS(app)                    # Allow frontend (React/Vue/etc.) to call this API

# Load model ONCE when server starts (fast!)
model = load_model()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "🚀 Customer Churn Prediction API is LIVE!",
        "endpoints": {
            "GET /": "This message",
            "POST /predict": "Send JSON → get churn prediction"
        },
        "features_order": ["tenure", "monthly_charges", "total_charges", "complaints", "satisfaction"]
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # ✅ NEW INPUT (ONLY 3 FEATURES)
        input_data = pd.DataFrame([{
            "tenure": data.get("tenure"),
            "MonthlyCharges": data.get("monthly_charges"),
            "TotalCharges": data.get("total_charges")
        }])

        # prediction
        churn = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return jsonify({
            "churn": int(churn),
            "probability": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

        # === Validation & Preparation ===
        X = validate_and_prepare_input(data)

        # ====================== PREPROCESSING BLOCK (IF YOU NEED IT) ======================
        # Uncomment the lines below if you saved a scaler
        #
        # from utils import load_scaler
        # scaler = load_scaler()               # load once globally is better
        # X = scaler.transform(X)
        # ================================================================================

        # Make prediction
        churn = model.predict(X)[0]                    # 0 or 1
        probability = model.predict_proba(X)[0][1]     # probability of churn (class 1)

        return jsonify({
            "churn": int(churn),           # 0 = stay, 1 = churn
            "probability": round(float(probability), 4)
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({
            "error": "Server error during prediction",
            "details": str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)