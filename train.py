import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import joblib # Required for saving the preprocessor

# --- 1. Load the dataset ---
# Assuming 'flight_accidents_india_synthetic.csv' is in the same directory
df = pd.read_csv('flight_accidents_india_synthetic.csv')

print("Dataset loaded successfully. Info:")
df.info()
print("\nFirst 5 rows:")
print(df.head())

# --- 2. Prepare Data for Training ---
# Separate target variable 'Accident'
# Exclude identifier columns and other target-related columns that won't be used as features for prediction
X = df.drop(columns=['Accident', 'Accident_Severity', 'Fatalities', 'Injuries', 'Aircraft_Damage', 'Cause_Category', 'Flight_ID', 'Date', 'Time_UTC'])
y = df['Accident']

# Identify numerical and categorical columns for preprocessing
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

print(f"\nNumerical columns identified: {numerical_cols}")
print(f"Categorical columns identified: {categorical_cols}")

# --- Updated Preprocessor ---
from sklearn.preprocessing import OneHotEncoder  # make sure this import is visible

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)

import joblib
joblib.dump(preprocessor, "preprocessor.pkl")
print("✅ Preprocessor updated and saved as 'preprocessor.pkl'")

# --- 3. Create Preprocessing Pipeline ---
# StandardScaler for numerical features
numerical_transformer = StandardScaler()

# OneHotEncoder for categorical features (handle_unknown='ignore' for unseen categories in new data)
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a ColumnTransformer to apply different transformers to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Fit and transform the training data
# This step applies scaling and one-hot encoding
X_processed = preprocessor.fit_transform(X)

print(f"\nShape of preprocessed data (X_processed): {X_processed.shape}")

# Reshape X_processed for Conv1D input: (samples, timesteps, features)
# For tabular data with Conv1D, each feature is treated as a timestep, and we add a channel dimension of 1.
X_processed = X_processed.reshape(X_processed.shape[0], X_processed.shape[1], 1)

print(f"Reshaped X_processed for Conv1D input: {X_processed.shape}")

# Split the dataset into training and testing sets
# stratify=y ensures that the proportion of 'Accident' (1s) is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- 4. Define and Compile the CNN Model ---
model = Sequential([
    # First Convolutional Layer
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2), # Reduces dimensionality

    # Second Convolutional Layer
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),

    # Flatten the output for the Dense layers
    Flatten(),

    # Fully Connected (Dense) Layers
    Dense(128, activation='relu'),
    Dropout(0.5), # Dropout for regularization to prevent overfitting

    # Output layer for binary classification
    Dense(1, activation='sigmoid') # Sigmoid activation for probabilities between 0 and 1
])

# Compile the model
# Adam optimizer is a good default choice
# binary_crossentropy for binary classification problems
# metrics=['accuracy'] to monitor performance during training
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

print("\nCNN Model Summary:")
model.summary()

# --- 5. Train the Model ---
print("\nStarting model training...")
history = model.fit(
    X_train, y_train,
    epochs=50,          # Number of training iterations
    batch_size=32,      # Number of samples per gradient update
    validation_split=0.2, # Use 20% of training data for validation during training
    verbose=1           # Show training progress
)

# --- 6. Evaluate the Model ---
print("\nEvaluating the model on the test set...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- 7. Save the Trained Model and Preprocessor ---
# Save the Keras model in HDF5 format
model_filename = 'flight_accident_cnn_model.h5'
model.save(model_filename)
print(f"\nTrained CNN model saved as '{model_filename}'")

# Save the preprocessor using joblib
# This is crucial because new data must be transformed using the exact same scaling and encoding
preprocessor_filename = 'preprocessor.pkl'
joblib.dump(preprocessor, preprocessor_filename)
print(f"Preprocessor saved as '{preprocessor_filename}'")

print("\nTraining and file creation process complete.")