import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd

# Load preprocessing pipeline and model
with open("preprocessing.pkl", "rb") as f:
    scaler, encoder = pickle.load(f)
model = tf.keras.models.load_model("tf_bridge_model.h5")

# Streamlit app
def main():
    st.title("Bridge Load Capacity Predictor")
    
    # User input fields
    span_ft = st.number_input("Span (ft)", min_value=10, max_value=1000, value=250)
    deck_width_ft = st.number_input("Deck Width (ft)", min_value=5, max_value=200, value=40)
    age_years = st.number_input("Age (Years)", min_value=0, max_value=150, value=20)
    num_lanes = st.number_input("Number of Lanes", min_value=1, max_value=10, value=2)
    condition_rating = st.slider("Condition Rating", 1, 5, 3)
    material = st.selectbox("Material", ["Composite", "Concrete", "Steel"])
    
    # Preprocess input
    input_data = np.array([[span_ft, deck_width_ft, age_years, num_lanes, condition_rating]])
    input_data = scaler.transform(input_data)
    
    material_encoded = [0, 0]  # Default: Composite
    if material == "Concrete":
        material_encoded = [1, 0]
    elif material == "Steel":
        material_encoded = [0, 1]
    
    input_final = np.hstack([input_data, material_encoded])
    
    # Prediction
    if st.button("Predict Load Capacity"):
        prediction = model.predict(input_final)
        st.success(f"Estimated Max Load Capacity: {prediction[0][0]:.2f} tons")

if __name__ == "__main__":
    main()
