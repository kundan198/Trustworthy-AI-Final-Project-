import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import defaultdict

# Load dataset
landmark_df = pd.read_csv("landmarks_data.csv")
demographic_data = pd.read_csv("demographic_data.csv")  # CSV with demographic info like 'skin_tone'

# Ensure matching data sizes
if len(landmark_df) != len(demographic_data):
    min_len = min(len(landmark_df), len(demographic_data))
    landmark_df = landmark_df.iloc[:min_len].reset_index(drop=True)
    demographic_data = demographic_data.iloc[:min_len].reset_index(drop=True)

# Separate features and labels
X = landmark_df.iloc[:, :-1].values  # Hand landmarks
y = landmark_df.iloc[:, -1].values   # Labels

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset including demographic info
X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
    X, y, demographic_data, test_size=0.2, random_state=42
)

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

# Save model and label encoder
model.save("sign_language_model.keras")
np.save("label_classes.npy", label_encoder.classes_)

print("Model training complete! ✅")

# Prediction
y_pred = np.argmax(model.predict(X_test), axis=1)

# Fairness evaluation
def calculate_fairness_metrics(y_true, y_pred, group_column):
    groups = np.unique(group_column)
    metrics = defaultdict(dict)

    for group in groups:
        group_indices = np.where(group_column == group)[0]
        y_true_group = y_true[group_indices]
        y_pred_group = y_pred[group_indices]

        metrics[group]['accuracy'] = accuracy_score(y_true_group, y_pred_group)
        metrics[group]['precision'] = precision_score(y_true_group, y_pred_group, average='macro', zero_division=0)
        metrics[group]['recall'] = recall_score(y_true_group, y_pred_group, average='macro', zero_division=0)

    return metrics

# Compute fairness metrics
fairness_results = calculate_fairness_metrics(y_test, y_pred, demo_test['skin_tone'].values)

print("\nFairness Evaluation Results:")
for group, metrics in fairness_results.items():
    print(f"Group: {group}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.capitalize()}: {value:.4f}")