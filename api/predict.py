# app.py - COMPLETE Flask Backend
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_model, validate_and_prepare_input

app = Flask(__name__)
CORS(app)

# Load model ONCE when server starts (fast!)
# Wrap in try-except in case model file doesn't exist
try:
    model = load_model()
except Exception as e:
    print(f"⚠️ Warning: Model could not be loaded at startup: {e}")
    model = None

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
        if model is None:
            return jsonify({"error": "Model not loaded. Server not ready."}), 503
            
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # === Validation & Preparation ===
        X = validate_and_prepare_input(data)

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