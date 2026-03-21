import os
import joblib
import numpy as np

def load_model():
    """Load the saved model once when app starts"""
    # Use path relative to this file's location for Vercel compatibility
    model_path = os.path.join(
        os.path.dirname(__file__),
        'model',
        'churn_model.pkl'
    )
    return joblib.load(model_path)

# ====================== PREPROCESSING (IF NEEDED) ====================== 
# Uncomment and use ONLY if you used scaling/encoding during training

# def load_scaler():
#     """Load scaler if you saved it separately"""
#     scaler_path = os.path.join(
#         os.path.dirname(__file__),
#         'model',
#         'scaler.pkl'
#     )
#     return joblib.load(scaler_path)

# def load_encoder():
#     """If you have any categorical features (none in your case)"""
#     pass
# ====================================================================== 

def validate_and_prepare_input(data: dict):
    """Validate input + return numpy array in EXACT feature order"""
    required_fields = ['tenure', 'monthly_charges', 'total_charges', 'complaints', 'satisfaction']
    
    # Check all fields exist
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    try:
        # Convert and keep STRICT order (same as training!)
        features = [
            float(data['tenure']),           # 0
            float(data['monthly_charges']),  # 1
            float(data['total_charges']),    # 2
            float(data['complaints']),       # 3 (can be 0,1,2...)
            float(data['satisfaction'])      # 4 (e.g. 1-5 or 0-1)
        ]
        return np.array([features])   # shape (1, 5) → what sklearn expects
    except (ValueError, TypeError):
        raise ValueError("Invalid data types. All fields must be numbers.")