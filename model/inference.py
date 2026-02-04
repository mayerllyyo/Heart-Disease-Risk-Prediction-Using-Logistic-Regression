import json
import numpy as np
import os

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'heart_disease_model.json')
    
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    return model_data

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return np.array(input_data['features'])
    else:
        raise ValueError(f"Content type {request_content_type} not supported")

def predict_fn(input_data, model):
    weights = np.array(model['weights'])
    bias = model['bias']
    mean = np.array(model['normalization_params']['mean'])
    std = np.array(model['normalization_params']['std'])
    
    input_norm = (input_data - mean) / std
    
    z = np.dot(input_norm, weights) + bias
    probability = sigmoid(z)
    
    return probability

def output_fn(prediction, accept):
    if accept == 'application/json':
        response = {
            'probability': float(prediction),
            'prediction': 1 if prediction >= 0.5 else 0,
            'risk_level': 'HIGH RISK' if prediction >= 0.5 else 'LOW RISK'
        }
        return json.dumps(response)
    else:
        raise ValueError(f"Accept type {accept} not supported")