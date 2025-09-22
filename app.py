import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, time

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and preprocessing tools
@st.cache_resource
def load_models():
    try:
        model = load_model("fraud_detection_mlp.h5")
        scaler = joblib.load("scaler.joblib")
        label_encoders = joblib.load("label_encoders.joblib")
        feature_names = joblib.load("feature_names.joblib")
        threshold = joblib.load("optimal_threshold.joblib")
        
        return model, scaler, label_encoders, feature_names, threshold
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

# Predefined values for categorical variables
category_options = [
    'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 
    'grocery_pos', 'health_fitness', 'home', 'kids_pets', 'misc_net', 
    'misc_pos', 'personal_care', 'shopping_net', 'shopping_pos', 'travel'
]

gender_options = ['F', 'M']

# Job options list (kısaltılmış versiyon)
job_options = ['Psychologist, counselling', 'Teacher', 'Engineer', 'Doctor', 'Nurse', 
               'Manager', 'Analyst', 'Developer', 'Consultant', 'Sales', 'Artist', 
               'Scientist', 'Technician', 'Accountant', 'Lawyer', 'Architect']

# Feature engineering functions
def calculate_distance(lat, long, merch_lat, merch_long):
    return np.sqrt((lat - merch_lat)**2 + (long - merch_long)**2)

def time_to_unix_hour(selected_time):
    """Convert time to hour of day (0-23)"""
    return selected_time.hour

def calculate_amt_per_pop(amt, city_pop):
    return amt / (city_pop + 1)

# Prediction function
def predict_fraud(model, input_data, threshold):
    try:
        prediction_proba = model.predict(input_data, verbose=0).flatten()[0]
        prediction = 1 if prediction_proba >= threshold else 0
        return prediction, prediction_proba
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Main application
def main():
    st.title("💰 Fraud Detection System")
    st.markdown("""
    This application uses AI models to detect credit card transaction fraud. 
    Please fill out the form below to analyze a transaction.
    
    **Note:** The model was trained on a highly imbalanced dataset (only 0.58% fraud).
    Therefore, it is optimized for high precision.
    """)
    
    # Load models
    model, scaler, label_encoders, feature_names, threshold = load_models()
    
    if model is None:
        st.stop()
    
    # Sidebar - model information
    with st.sidebar:
        st.header("Model Information")
        st.info(f"Model: **Neural Network**\n\nThreshold: **{threshold:.4f}**")
        
        st.header("Dataset Statistics")
        st.markdown("""
        - Total Transactions: 1,296,675
        - Legitimate Transactions: 1,289,169 (99.42%)
        - Fraud Transactions: 7,506 (0.58%)
        - Imbalance Ratio: 172:1
        """)
        
        st.header("About")
        st.markdown("""
        This fraud detection system uses the following techniques:
        - Deep Neural Networks (MLP)
        - SMOTE + Tomek Links for data balancing
        - Advanced feature engineering
        """)
    
    # Main content - input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        amt = st.number_input("Transaction Amount (USD)", min_value=0.0, max_value=10000.0, value=50.0, step=0.01)
        category = st.selectbox("Category", category_options)
        
        # Sadece saat girişi
        st.subheader("Transaction Time")
        trans_time = st.time_input("Transaction Time", value=time(12, 0))
        
        # Saati hour feature'ına çevir (0-23)
        hour = time_to_unix_hour(trans_time)
        
        st.write(f"**Transaction Hour:** {hour}:00")
        
        # Unix time için sabit bir değer kullan (modelin beklediği formatı korumak için)
        # Gerçek uygulamada bu değer önemli değil çünkü model hour feature'ını kullanıyor
        unix_time = 1325376000 + (hour * 3600)  # Sabit bir gün + saat offset
        
        st.subheader("Customer Information")
        gender = st.selectbox("Gender", gender_options)
        lat = st.number_input("Customer Latitude", min_value=-90.0, max_value=90.0, value=40.7128, format="%.6f")
        long = st.number_input("Customer Longitude", min_value=-180.0, max_value=180.0, value=-74.0060, format="%.6f")
        city_pop = st.number_input("City Population", min_value=0, value=100000)
        
        # Job selection with dropdown
        job = st.selectbox("Job", job_options)
    
    with col2:
        st.subheader("Merchant Information")
        merch_lat = st.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, value=40.7210, format="%.6f")
        merch_long = st.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, value=-74.0090, format="%.6f")
        
        # Show feature engineering results
        st.subheader("Calculated Features")
        distance = calculate_distance(lat, long, merch_lat, merch_long)
        st.write(f"**Customer-Merchant Distance:** {distance:.2f} units")
        
        st.write(f"**Transaction Hour:** {hour:02d}:00")
        
        amt_per_pop = calculate_amt_per_pop(amt, city_pop)
        st.write(f"**Amount Normalized by Population:** {amt_per_pop:.8f}")
        
        # Transaction summary
        st.subheader("Transaction Summary")
        st.write(f"**Time:** {trans_time.strftime('%H:%M:%S')}")
        st.write(f"**Amount:** ${amt:.2f}")
        st.write(f"**Category:** {category}")
        st.write(f"**Customer Location:** ({lat:.4f}, {long:.4f})")
        st.write(f"**Merchant Location:** ({merch_lat:.4f}, {merch_long:.4f})")
        st.write(f"**Distance:** {distance:.2f} units")
    
    # Prediction button
    if st.button("Analyze Transaction", type="primary"):
        # Preprocess data
        try:
            # Encode categorical variables
            category_encoded = label_encoders['category'].transform([category])[0]
            gender_encoded = label_encoders['gender'].transform([gender])[0]
            
            # Job encoding - use 'unknown' for unknown values
            if job in label_encoders['job'].classes_:
                job_encoded = label_encoders['job'].transform([job])[0]
            else:
                job_encoded = label_encoders['job'].transform(['unknown'])[0]
                st.warning(f"Job '{job}' not found in training data. Marked as 'unknown'.")
            
            # Create feature vector (modelin beklediği sıraya göre)
            input_features = np.array([[
                amt, category_encoded, gender_encoded, lat, long, 
                city_pop, job_encoded, merch_lat, merch_long, unix_time,
                distance, hour, amt_per_pop
            ]])
            
            # Scale features
            input_scaled = scaler.transform(input_features)
            
            # Make prediction
            prediction, prediction_proba = predict_fraud(model, input_scaled, threshold)
            
            # Check if prediction was successful
            if prediction is None or prediction_proba is None:
                st.error("Failed to make prediction. Please check your input data and try again.")
                return
            
            # Display results
            st.subheader("🎯 Analysis Results")
            
            # Create columns for results display
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                # Determine risk level
                if prediction_proba < 0.3:
                    risk_level = "Low Risk"
                    color = "green"
                    emoji = "✅"
                elif prediction_proba < 0.7:
                    risk_level = "Medium Risk"
                    color = "orange"
                    emoji = "⚠️"
                else:
                    risk_level = "High Risk"
                    color = "red"
                    emoji = "🚨"
                
                if prediction == 1:
                    st.error(f"{emoji} **FRAUD TRANSACTION ALERT**")
                else:
                    st.success(f"{emoji} **NORMAL TRANSACTION**")
                
                st.markdown(f"**Risk Level:** <span style='color:{color};font-size:20px'>{risk_level}</span>", 
                           unsafe_allow_html=True)
                st.markdown(f"**Fraud Probability:** `{prediction_proba:.4f}`")
                st.markdown(f"**Decision Threshold:** `{threshold:.4f}`")
            
            with res_col2:
                # Probability gauge
                st.markdown("**Risk Probability Gauge:**")
                st.progress(float(prediction_proba))
                
                # Additional metrics
                st.markdown("**Confidence Levels:**")
                if prediction_proba < 0.1:
                    st.info("Very Low Risk (0-10%)")
                elif prediction_proba < 0.3:
                    st.info("Low Risk (10-30%)")
                elif prediction_proba < 0.5:
                    st.warning("Moderate Risk (30-50%)")
                elif prediction_proba < 0.7:
                    st.warning("Elevated Risk (50-70%)")
                else:
                    st.error("High Risk (70-100%)")
            
            # Recommendation based on result
            st.markdown("---")
            if prediction == 1:
                st.error("""
                **🔍 Recommended Actions:**
                - Conduct additional verification
                - Contact customer for confirmation
                - Review transaction patterns
                - Consider temporary hold if necessary
                """)
            else:
                st.success("""
                **✅ Transaction Status: CLEAR**
                - No additional verification required
                - Standard processing recommended
                """)
            
            # Model performance info
            st.info(f"**Model:** Neural Network | **Threshold:** {threshold:.4f} | **Processing Time:** Real-time")
            
            # Statistical context
            with st.expander("📊 Statistical Context"):
                st.markdown("""
                **Dataset Statistics:**
                - Fraud rate in training data: 0.58%
                - Model optimized for high precision
                - Focus on minimizing false positives
                
                **Interpretation Guide:**
                - **< 0.3:** Low risk - Normal processing
                - **0.3-0.7:** Medium risk - Review recommended  
                - **> 0.7:** High risk - Immediate action suggested
                """)
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            st.info("Please check that all input values are valid and try again.")

# Run the application
if __name__ == "__main__":
    main()