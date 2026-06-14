"""
train_model.py
Loads the gesture landmark dataset, trains a RandomForestClassifier,
evaluates it, and saves the trained model and label encoder.
"""

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CSV_PATH = "data/gestures.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{CSV_PATH}'. "
        f"Run data_collection.py first to collect gesture samples."
    )

df = pd.read_csv(CSV_PATH)

if df.empty:
    raise ValueError(
        f"Dataset at '{CSV_PATH}' is empty. "
        f"Collect samples using data_collection.py before training."
    )

print(f"Loaded dataset with {len(df)} samples and {df['label'].nunique()} classes.")
print("Class distribution:")
print(df['label'].value_counts())

# ---------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------
# Drop rows with any missing values
df = df.dropna()

# Ensure all feature columns are numeric
feature_columns = [col for col in df.columns if col != "label"]
df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
df = df.dropna()

if df.empty:
    raise ValueError("Dataset is empty after cleaning. Please collect more data.")

# ---------------------------------------------------------
# FEATURES AND LABELS
# ---------------------------------------------------------
X = df[feature_columns].values
y_raw = df["label"].values

# ---------------------------------------------------------
# LABEL ENCODING
# ---------------------------------------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print(f"\nClasses found: {list(label_encoder.classes_)}")

# ---------------------------------------------------------
# TRAIN/TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=[str(c) for c in label_encoder.classes_],
    zero_division=0
))

# ---------------------------------------------------------
# SAVE MODEL AND LABEL ENCODER
# ---------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Label encoder saved to: {ENCODER_PATH}")