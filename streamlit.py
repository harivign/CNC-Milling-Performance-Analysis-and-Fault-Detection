# import streamlit as st
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# # Load the saved model
# model = load_model('lstm_model.h5')

# import joblib

# # Load the scaler
# scaler = joblib.load('scaler.pkl')

# # Load the label encoders
# label_encoders = joblib.load('label_encoders.pkl')

# # Load the scaler (you need to save it during training)
# scaler = StandardScaler()

# # Load label encoders (you need to save them during training)
# label_encoders = {
#     'tool_condition': LabelEncoder().fit(['unworn', 'worn']),
#     'machining_finalized': LabelEncoder().fit(['yes', 'no']),
#     'passed_visual_inspection': LabelEncoder().fit(['yes', 'no'])
# }

# # Function to preprocess the input data
# def preprocess_data(df):
#     # Drop unnecessary columns (if any)
#     df = df.drop(columns=['No', 'material', 'feedrate', 'clamp_pressure'], errors='ignore')
    
#     # Normalize/Standardize the features
#     X_scaled = scaler.transform(df)
    
#     # Reshape data for LSTM (samples, timesteps, features)
#     X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
#     return X_reshaped

# # Function to decode predictions
# def decode_predictions(predictions):
#     decoded_predictions = {
#         'tool_condition': label_encoders['tool_condition'].inverse_transform(np.argmax(predictions[0], axis=1)),
#         'machining_finalized': label_encoders['machining_finalized'].inverse_transform(np.argmax(predictions[1], axis=1)),
#         'passed_visual_inspection': label_encoders['passed_visual_inspection'].inverse_transform(np.argmax(predictions[2], axis=1))
#     }
#     return decoded_predictions

# # Streamlit app
# st.title("Tool Wear Prediction using LSTM")
# st.write("Upload a CSV file to predict tool condition, machining finalized, and visual inspection status.")

# # File upload
# uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# if uploaded_file is not None:
#     # Read the uploaded file
#     df = pd.read_csv(uploaded_file)
    
#     # Display the uploaded data
#     st.write("Uploaded Data:")
#     st.write(df)
    
#     # Preprocess the data
#     try:
#         X_processed = preprocess_data(df)
        
#         # Make predictions
#         predictions = model.predict(X_processed)
        
#         # Decode predictions
#         decoded_predictions = decode_predictions(predictions)
        
#         # Display predictions
#         st.write("Predictions:")
#         predictions_df = pd.DataFrame(decoded_predictions)
#         st.write(predictions_df)
        
#         # Option to download predictions
#         st.download_button(
#             label="Download Predictions",
#             data=predictions_df.to_csv(index=False).encode('utf-8'),
#             file_name='predictions.csv',
#             mime='text/csv'
#         )
#     except Exception as e:
#         st.error(f"Error processing the file: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ✅ Load the saved LSTM model
try:
    model = load_model('lstm_model.h5')
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ✅ Load the saved scaler
try:
    scaler = joblib.load('scaler.pkl')
    st.success("✅ Scaler loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading scaler: {e}")
    st.stop()

# ✅ Load label encoders (ensure these were saved during training)
try:
    label_encoders = joblib.load('label_encoders.pkl')
    st.success("✅ Label Encoders loaded successfully!")
except Exception:
    # If label_encoders.pkl is not found, define manually
    label_encoders = {
        'tool_condition': LabelEncoder().fit(['unworn', 'worn']),
        'machining_finalized': LabelEncoder().fit(['yes', 'no']),
        'passed_visual_inspection': LabelEncoder().fit(['yes', 'no'])
    }

# ✅ Define required columns
required_columns = [
    'X1_ActualPosition', 'X1_ActualVelocity', 'X1_ActualAcceleration',
    'X1_CommandPosition', 'X1_CommandVelocity', 'X1_CommandAcceleration',
    'X1_CurrentFeedback', 'X1_DCBusVoltage', 'X1_OutputCurrent',
    'X1_OutputVoltage', 'X1_OutputPower', 'Y1_ActualPosition',
    'Y1_ActualVelocity', 'Y1_ActualAcceleration', 'Y1_CommandPosition',
    'Y1_CommandVelocity', 'Y1_CommandAcceleration', 'Y1_CurrentFeedback',
    'Y1_DCBusVoltage', 'Y1_OutputCurrent', 'Y1_OutputVoltage',
    'Y1_OutputPower', 'Z1_ActualPosition', 'Z1_ActualVelocity',
    'Z1_ActualAcceleration', 'Z1_CommandPosition', 'Z1_CommandVelocity',
    'Z1_CommandAcceleration', 'S1_ActualPosition', 'S1_ActualVelocity',
    'S1_ActualAcceleration', 'S1_CommandPosition', 'S1_CommandVelocity',
    'S1_CommandAcceleration', 'S1_CurrentFeedback', 'S1_DCBusVoltage',
    'S1_OutputCurrent', 'S1_OutputVoltage', 'S1_OutputPower',
    'M1_CURRENT_PROGRAM_NUMBER', 'M1_sequence_number',
    'M1_CURRENT_FEEDRATE', 'Encoded_material', 'Encoded_feedrate',
    'Encoded_clamp_pressure'
]

# ✅ Function to preprocess input data
def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=['No', 'material', 'feedrate', 'clamp_pressure'], errors='ignore')

    # Ensure all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"⚠️ Missing columns: {missing_columns}. Please upload the correct format.")
        st.stop()

    # Keep only required columns
    df = df[required_columns]

    # Normalize features
    X_scaled = scaler.transform(df)

    # Reshape data for LSTM (samples, timesteps, features)
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    return X_reshaped

# ✅ Function to decode model predictions
def decode_predictions(predictions):
    # Convert probabilities to binary outputs (0 or 1)
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)

    decoded_predictions = {
        'Tool Condition': label_encoders['tool_condition'].inverse_transform(binary_predictions[:, 0]),
        'Machining Finalized': label_encoders['machining_finalized'].inverse_transform(binary_predictions[:, 1]),
        'Passed Visual Inspection': label_encoders['passed_visual_inspection'].inverse_transform(binary_predictions[:, 2])
    }
    return pd.DataFrame(decoded_predictions)

# ✅ Streamlit UI
st.title("🔧 Tool Wear Prediction using LSTM")
st.write("Upload a **CSV file** to predict tool condition, machining finalization, and visual inspection status.")

# File Upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    # Display uploaded data
    st.write("📊 Uploaded Data Preview:")
    st.dataframe(df.head())

    # Preprocess the data
    try:
        X_processed = preprocess_data(df)

        # Make predictions
        predictions = model.predict(X_processed)

        # Decode predictions
        results_df = decode_predictions(predictions)

        # Display results
        st.write("🔍 **Predictions:**")
        st.dataframe(results_df)

        # Allow downloading of results
        csv_output = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="⬇️ Download Predictions as CSV", data=csv_output, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error processing the file: {e}")
