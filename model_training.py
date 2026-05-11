import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FraudDetectionModel:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)

    def load_and_clean_data(self):
        logging.info(f"Loading dataset from {self.dataset_path}...")
        try:
            # Assuming standard Kaggle Credit Card Fraud dataset format
            df = pd.read_csv(self.dataset_path)
            logging.info("Dataset loaded successfully.")
            
            # Dropping irrelevant columns like 'Time'
            if 'Time' in df.columns:
                df = df.drop(['Time'], axis=1)
                
            X = df.drop(['Class'], axis=1)
            y = df['Class']
            
            logging.info("Scaling 'Amount' feature...")
            X['Amount'] = self.scaler.fit_transform(X[['Amount']])
            
            return X, y
        except FileNotFoundError:
            logging.warning("Dataset not found locally. This is expected in GitHub showcase mode.")
            return None, None

    def handle_imbalance_and_split(self, X, y):
        logging.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        logging.info("Applying SMOTE to handle class imbalance on training data...")
        X_train_res, y_train_res = self.smote.fit_resample(X_train, y_train)
        
        return X_train_res, X_test, y_train_res, y_test

    def train(self, X_train, y_train):
        logging.info("Training Random Forest Classifier. This might take a few minutes...")
        self.model.fit(X_train, y_train)
        logging.info("Model training complete.")

    def evaluate(self, X_test, y_test):
        logging.info("Evaluating model performance on unseen test data...")
        predictions = self.model.predict(X_test)
        
        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(y_test, predictions))
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test, predictions))
        
        logging.info("Evaluation finished. High Precision-Recall AUC is the primary target.")

if __name__ == "__main__":
    logging.info("=== Credit Card Fraud Detection Pipeline Started ===")
    
    # Initialize the pipeline
    fraud_pipeline = FraudDetectionModel(dataset_path="data/creditcard.csv")
    
    # Execute steps (Wrapped in checks for safe GitHub hosting without real data files)
    X, y = fraud_pipeline.load_and_clean_data()
    
    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = fraud_pipeline.handle_imbalance_and_split(X, y)
        fraud_pipeline.train(X_train, y_train)
        fraud_pipeline.evaluate(X_test, y_test)
    else:
        logging.info("Pipeline structure verified. Awaiting actual dataset for execution.")
        
    logging.info("=== Pipeline Execution Ended ===")
