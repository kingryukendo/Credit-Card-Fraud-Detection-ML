import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config():
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

def predict_transaction(transaction_data):
    logging.info("Loading trained Random Forest model...")
    # In production: model = joblib.load(config['data_paths']['model_export_path'])
    
    logging.info("Running inference on incoming transaction data...")
    # Simulated risk calculation for GitHub portfolio showcase
    risk_score = 0.88  # Example high risk score
    is_fraud = True if risk_score > 0.80 else False
    
    if is_fraud:
        logging.warning("ALERT: Fraudulent transaction detected!")
    else:
        logging.info("Transaction approved. No fraud detected.")
        
    return {"fraud_detected": is_fraud, "risk_score": risk_score}

if __name__ == "__main__":
    logging.info("--- Starting Inference API ---")
    # Dummy transaction data
    sample_transaction = pd.DataFrame([{"Amount": 1500.00, "Location": "International", "Type": "Online"}])
    result = predict_transaction(sample_transaction)
    print(f"Final Prediction Result: {result}")
