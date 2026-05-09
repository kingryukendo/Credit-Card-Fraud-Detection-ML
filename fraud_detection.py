import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset (Kaggle Credit Card Fraud Data)
# Note: Ensure creditcard.csv is in the same directory
def train_fraud_model(data_path):
    df = pd.read_csv(data_path)
    
    # Preprocessing
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Model Initialization (Random Forest)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    print("Credit Card Fraud Detection Model Training...")
    # train_fraud_model('creditcard.csv') # Uncomment to run with actual data
