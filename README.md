# 🛡️ Fraud Detection with Neural Network & Advanced Feature Engineering

This project implements a **Fraud Detection System** using a **Neural Network (Keras/TensorFlow)**, enriched with **feature engineering, class balancing techniques, and threshold optimization**. It is designed to handle **highly imbalanced datasets** and maximize fraud detection accuracy with real-world applicability.

## Dataset

The dataset used in this project is the [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) available on Kaggle. It contains simulated credit card transactions generated using Sparkov, covering the period from January 1, 2019, to December 31, 2020.

---

## 📂 Project Structure
├── fraudTrain.csv # Training dataset
├── fraudTest.csv # Testing dataset
├── fraud_detection_mlp.h5 # Saved Neural Network model
├── scaler.joblib # StandardScaler for numerical features
├── label_encoders.joblib # Encoders for categorical features
├── optimal_threshold.joblib # Optimal classification threshold
├── feature_names.joblib # Saved feature names
├── fraud_detection_ensemble.joblib (optional) # Placeholder for ensemble model
└── fraud_detection.ipynb # Main training & evaluation script

---

## 🚀 Features
- **Data Preprocessing**
  - Removes unnecessary identifiers (`trans_num`, `cc_num`, etc.)
  - Encodes categorical variables (`category`, `gender`, `job`)
  - Handles unknown categories gracefully
  - Applies `StandardScaler` for normalization
- **Feature Engineering**
  - Distance between customer & merchant (`distance`)
  - Transaction time as hour of day (`hour`)
  - Transaction amount normalized by population (`amt_per_pop`)
- **Imbalanced Data Handling**
  - Uses **SMOTE + Tomek Links** for resampling  
  - Computes **class weights** for fairness in training
- **Modeling**
  - Deep Neural Network with:
    - Batch Normalization
    - Dropout layers
    - Adaptive learning rate (`ReduceLROnPlateau`)
    - Early stopping to prevent overfitting
  - Optimized **decision threshold** based on F1-score
- **Evaluation Metrics**
  - ROC-AUC, PR-AUC, Average Precision
  - Matthews Correlation Coefficient (MCC)
  - Fraud-specific metrics (Precision, Recall, F1 for fraud class)
  - Confusion Matrix visualization
- **Outputs**
  - Model performance charts (Loss, Accuracy, Precision-Recall)
  - Saved models & preprocessing artifacts for deployment
