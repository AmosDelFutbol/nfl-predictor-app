import pickle
import pandas as pd
import numpy as np

# Load and inspect the model
try:
    with open('nfl_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    print("=== MODEL DIAGNOSTICS ===")
    print(f"Model type: {type(model_data)}")
    print(f"Model contents: {model_data}")
    
    if isinstance(model_data, dict):
        print("\n=== DICTIONARY STRUCTURE ===")
        for key, value in model_data.items():
            print(f"{key}: {type(value)} - {value if not isinstance(value, (list, np.ndarray)) or len(value) < 10 else f'array with {len(value)} elements'}")
            
        # Check if it has weights for prediction
        if 'weights' in model_data:
            weights = model_data['weights']
            print(f"\nWeights shape: {np.array(weights).shape if hasattr(weights, '__len__') else 'scalar'}")
            print(f"Weights sum: {np.sum(weights) if hasattr(weights, '__len__') else weights}")
            
    elif hasattr(model_data, 'feature_importances_'):
        print(f"\nFeature importances: {model_data.feature_importances_}")
        
except Exception as e:
    print(f"Error loading model: {e}")
