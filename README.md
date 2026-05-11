# Credit Card Fraud Detection ML Pipeline

An enterprise-grade Machine Learning pipeline designed to detect fraudulent credit card transactions with high precision, handling highly imbalanced datasets.

## 🚀 Project Overview
Fraudulent transactions make up a microscopic fraction of global credit card usage. This project implements a robust anomaly detection and classification pipeline using advanced ensemble methods (Random Forest) to accurately identify fraud without causing high false-positive rates for legitimate users.

## 🧠 Technical Architecture
- **Data Preprocessing:** Feature scaling, handling missing values, and dimensionality reduction (PCA).
- **Class Imbalance Handling:** Implemented strategic sampling techniques to train unbiased models.
- **Model Training:** Utilized `RandomForestClassifier` for robust non-linear decision boundaries.
- **Evaluation Metrics:** Focused strictly on **Precision-Recall AUC** and **F1-Score** rather than plain accuracy.

## 🛠️ Tech Stack
- **Language:** Python 3.9+
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Logging & Tracking:** Python Logging Module

## 📊 Business Impact
By minimizing false positives, this model ensures that legitimate customer transactions are not blocked, preserving user experience while saving millions in potential fraud losses.
