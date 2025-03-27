import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle

# Load dataset
df = pd.read_csv("lab_11_bridge_data.csv")

# Drop Bridge_ID
df.drop(columns=['Bridge_ID'], inplace=True)

# One-hot encode Material
encoder = OneHotEncoder(sparse=False, drop='first')
material_encoded = encoder.fit_transform(df[['Material']])
material_cols = encoder.get_feature_names_out(['Material'])
df_encoded = pd.DataFrame(material_encoded, columns=material_cols)

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['Span_ft', 'Deck_Width_ft', 'Age_Years', 'Num_Lanes', 'Condition_Rating']
df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)

# Combine processed data
df_final = pd.concat([df_scaled, df_encoded, df[['Max_Load_Tons']]], axis=1)

# Train-test split
X = df_final.drop(columns=['Max_Load_Tons'])
y = df_final['Max_Load_Tons']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessing pipeline
with open("preprocessing.pkl", "wb") as f:
    pickle.dump((scaler, encoder), f)

# Build ANN model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train model
model = build_model()
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])

# Save trained model
model.save("tf_bridge_model.h5")

# Evaluate model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae}")
