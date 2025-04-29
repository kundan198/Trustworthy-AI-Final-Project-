import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset with demographic info
df = pd.read_csv("landmarks_data_with_demographics.csv")

# Separate features and labels
X = df.iloc[:, :-3].values  # Hand landmarks (excluding skin_tone and hand_shape)
y = df.iloc[:, -3].values   # Labels (excluding skin_tone and hand_shape)
demographic_info = df[['skin_tone', 'hand_shape']]  # Demographic info (skin_tone, hand_shape)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(X, y, demographic_info, test_size=0.2, random_state=42)

# Define model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(set(y)), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save model & labels
model.save("sign_language_model.h5")
np.save("label_classes.npy", label_encoder.classes_)

print("Model training complete! ✅")

# Evaluate fairness and robustness after training
# Function to calculate fairness metrics
def calculate_fairness_metrics(y_true, y_pred, demographic_info):
    fairness_results = {}
    for group in demographic_info.unique():
        group_indices = demographic_info[demographic_info == group].index
        y_true_group = y_true[group_indices]
        y_pred_group = y_pred[group_indices]
        group_accuracy = accuracy_score(y_true_group, y_pred_group)
        fairness_results[group] = group_accuracy
    return fairness_results

# Get predictions
y_pred = model.predict(X_test)

# Fairness evaluation
fairness_results = calculate_fairness_metrics(y_test, y_pred, demo_test['skin_tone'])
print(f"Fairness results across skin tones: {fairness_results}")
