
# ============================================================
# ESC-50 PROJECT
# PHASE 4: MFCC + LOGISTIC REGRESSION BASELINE
# ============================================================

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

AUDIO_DIR = PROJECT_DIR / "audio"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

train_df = pd.read_csv(RESULTS_DIR / "train_split.csv")
val_df = pd.read_csv(RESULTS_DIR / "validation_split.csv")
test_df = pd.read_csv(RESULTS_DIR / "test_split.csv")

# ------------------------------------------------------------
# 2. AUDIO FEATURE EXTRACTION
# ------------------------------------------------------------

def extract_mfcc(filename):

    audio_path = AUDIO_DIR / filename

    # Load audio at 22050 Hz
    audio, sr = librosa.load(
        audio_path,
        sr=22050,
        mono=True
    )

    # Extract 40 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40,
        n_fft=2048,
        hop_length=512
    )

    # Calculate mean and standard deviation
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Combine into 80 numerical features
    features = np.concatenate([
        mfcc_mean,
        mfcc_std
    ])

    return features


# ------------------------------------------------------------
# 3. EXTRACT FEATURES FOR EACH DATASET
# ------------------------------------------------------------

def prepare_features(dataframe, name):

    print(f"\nExtracting features: {name}")

    X = np.array([
        extract_mfcc(filename)
        for filename in dataframe["filename"]
    ])

    y = dataframe["target"].to_numpy()

    print("Feature matrix shape:", X.shape)
    print("Label shape:", y.shape)

    return X, y


X_train, y_train = prepare_features(
    train_df, "Training"
)

X_val, y_val = prepare_features(
    val_df, "Validation"
)

X_test, y_test = prepare_features(
    test_df, "Testing"
)


# ------------------------------------------------------------
# 4. CHECK FEATURE MATRICES
# ------------------------------------------------------------

print("\nFEATURE VERIFICATION")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

assert X_train.shape[1] == 80
assert X_val.shape[1] == 80
assert X_test.shape[1] == 80

# ------------------------------------------------------------
# 5. BUILD LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

# StandardScaler prevents features with large
# numerical ranges from dominating the model.

# Logistic Regression learns relationships between
# the 80 features and 50 sound categories.

# ------------------------------------------------------------
# 6. SELECT HYPERPARAMETER USING VALIDATION DATA
# ------------------------------------------------------------

C_VALUES = [0.01, 0.1, 1, 10]

best_model = None
best_c = None
best_val_f1 = -1

print("\nMODEL TRAINING AND VALIDATION")
print("=" * 55)

for c in C_VALUES:

    model = Pipeline([
        ("scaler", StandardScaler()),

        ("classifier", LogisticRegression(
            C=c,
            max_iter=2000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)

    val_accuracy = accuracy_score(
        y_val, val_predictions
    )

    val_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro"
    )

    print(f"\nC = {c}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Macro F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_c = c
        best_model = model


# ------------------------------------------------------------
# 7. FINAL TEST EVALUATION
# ------------------------------------------------------------

# Test data is used only after selecting the model
# using validation performance.

test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test, test_predictions
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)

print("Best C:", best_c)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)


# ------------------------------------------------------------
# 8. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=np.arange(50)
)

class_names = (
    train_df
    .sort_values("target")["category"]
    .drop_duplicates()
    .tolist()
)

plt.figure(figsize=(20, 17))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("ESC-50 Logistic Regression Confusion Matrix")
plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "05_baseline_confusion_matrix.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 9. SAVE BASELINE RESULTS
# ------------------------------------------------------------

baseline_results = pd.DataFrame([{
    "Model": "MFCC + Logistic Regression",
    "Best_C": best_c,
    "Validation_Macro_F1": best_val_f1,
    "Test_Accuracy": test_accuracy,
    "Test_Macro_F1": test_f1
}])

baseline_results.to_csv(
    RESULTS_DIR / "baseline_results.csv",
    index=False
)

# ------------------------------------------------------------
# 10. SAVE TEST PREDICTIONS
# ------------------------------------------------------------

predictions_df = test_df[
    ["filename", "target", "category", "src_file"]
].copy()

predictions_df["predicted_target"] = test_predictions

predictions_df["predicted_category"] = [
    class_names[i] for i in test_predictions
]

predictions_df.to_csv(
    RESULTS_DIR / "baseline_test_predictions.csv",
    index=False
)

print("\nBaseline results saved successfully!")

print("\nPHASE 4 COMPLETED")
