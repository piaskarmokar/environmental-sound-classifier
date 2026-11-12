
# ============================================================
# PHASE 6: DETAILED ERROR ANALYSIS
# Dataset: ESC-50
# Model: CNN Mel Spectrogram
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

PREDICTIONS_FILE = (
    RESULTS_DIR / "cnn_test_predictions.csv"
)

TEST_FILE = (
    RESULTS_DIR / "test_split.csv"
)

RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("PHASE 6: CNN ERROR ANALYSIS")
print("=" * 65)


# ============================================================
# 2. LOAD PREDICTIONS AND TEST DATA
# ============================================================

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {PREDICTIONS_FILE}"
    )

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Test split file not found: {TEST_FILE}"
    )

predictions_df = pd.read_csv(PREDICTIONS_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nPrediction data shape:", predictions_df.shape)
print("Test data shape:", test_df.shape)

print("\nPrediction data preview:")
print(predictions_df.head())


# ============================================================
# 3. VERIFY THE PREDICTIONS
# ============================================================

required_columns = {
    "filename",
    "actual",
    "predicted"
}

if not required_columns.issubset(
    predictions_df.columns
):
    raise ValueError(
        "Prediction CSV must contain filename, actual, "
        "and predicted columns."
    )

if not predictions_df["filename"].is_unique:
    raise ValueError(
        "Duplicate filenames found in predictions."
    )

if set(predictions_df["filename"]) != set(test_df["filename"]):
    raise ValueError(
        "Prediction filenames do not match the test split."
    )

# Put predictions in exactly the same order as test_df

predictions_df = (
    test_df[["filename"]]
    .merge(
        predictions_df,
        on="filename",
        how="left",
        validate="one_to_one"
    )
)

if predictions_df[
    ["actual", "predicted"]
].isna().any().any():
    raise ValueError(
        "Missing actual or predicted labels detected."
    )

# Verify actual labels against the test metadata

check_df = predictions_df.merge(
    test_df[["filename", "category"]],
    on="filename",
    how="left",
    validate="one_to_one"
)

if not (
    check_df["actual"] == check_df["category"]
).all():
    raise ValueError(
        "Actual prediction labels do not match test metadata."
    )

print("\nPrediction verification successful.")


# ============================================================
# 4. OVERALL MODEL PERFORMANCE
# ============================================================

y_true = predictions_df["actual"]
y_pred = predictions_df["predicted"]

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\nOVERALL PERFORMANCE")
print("-" * 50)

print("Total test recordings:", len(predictions_df))
print("Correct predictions:", (y_true == y_pred).sum())
print("Incorrect predictions:", (y_true != y_pred).sum())

print("Test accuracy:", round(accuracy, 4))


# ============================================================
# 5. PER-CLASS PERFORMANCE
# ============================================================

class_names = sorted(
    test_df["category"].unique().tolist()
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_names,
        zero_division=0
    )
)

class_results = pd.DataFrame({

    "category": class_names,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "support": support

})

class_results["correct"] = (
    predictions_df.assign(
        correct=(
            predictions_df["actual"]
            == predictions_df["predicted"]
        )
    )
    .groupby("actual")["correct"]
    .sum()
    .reindex(class_names, fill_value=0)
    .values
)

class_results["incorrect"] = (
    class_results["support"]
    - class_results["correct"]
)

class_results = class_results.sort_values(
    "f1_score",
    ascending=True
)

print("\nPER-CLASS PERFORMANCE")
print("-" * 65)

print(
    class_results.to_string(
        index=False,
        formatters={
            "precision": "{:.3f}".format,
            "recall": "{:.3f}".format,
            "f1_score": "{:.3f}".format
        }
    )
)

class_results.to_csv(
    RESULTS_DIR / "cnn_per_class_error_analysis.csv",
    index=False
)


# ============================================================
# 6. IDENTIFY THE MOST CONFUSED SOUND CATEGORIES
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=class_names
)

confusion_records = []

for i, actual_class in enumerate(class_names):

    for j, predicted_class in enumerate(class_names):

        if i != j and cm[i, j] > 0:

            confusion_records.append({

                "actual_class": actual_class,

                "predicted_class": predicted_class,

                "number_of_errors": int(cm[i, j])

            })


confusion_pairs = pd.DataFrame(
    confusion_records,
    columns=[
        "actual_class",
        "predicted_class",
        "number_of_errors"
    ]
)

confusion_pairs = confusion_pairs.sort_values(
    "number_of_errors",
    ascending=False
)

print("\nMOST FREQUENT CLASSIFICATION ERRORS")
print("-" * 65)

if len(confusion_pairs) > 0:

    print(
        confusion_pairs.head(20).to_string(
            index=False
        )
    )

else:

    print("No classification errors found.")

confusion_pairs.to_csv(
    RESULTS_DIR / "cnn_most_confused_pairs.csv",
    index=False
)


# ============================================================
# 7. LIST INCORRECTLY CLASSIFIED RECORDINGS
# ============================================================

error_df = predictions_df[
    predictions_df["actual"]
    != predictions_df["predicted"]
].copy()

error_df["error_description"] = (
    error_df["actual"]
    + " predicted as "
    + error_df["predicted"]
)

print("\nINCORRECTLY CLASSIFIED RECORDINGS")
print("-" * 65)

print("Number of errors:", len(error_df))

print(
    error_df[
        [
            "filename",
            "actual",
            "predicted"
        ]
    ].head(30).to_string(index=False)
)

error_df.to_csv(
    RESULTS_DIR / "cnn_incorrect_predictions.csv",
    index=False
)


# ============================================================
# 8. NORMALIZED CONFUSION MATRIX
# ============================================================

# Each row is divided by its actual class support.
# This shows the proportion of each true class
# assigned to each predicted class.

row_totals = cm.sum(axis=1, keepdims=True)

cm_normalized = np.divide(
    cm,
    row_totals,
    out=np.zeros_like(cm, dtype=float),
    where=row_totals != 0
)

plt.figure(figsize=(19, 16))

sns.heatmap(
    cm_normalized,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    vmin=0,
    vmax=1
)

plt.title(
    "CNN Normalized Confusion Matrix",
    fontsize=16
)

plt.xlabel("Predicted Sound Category")
plt.ylabel("Actual Sound Category")

plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "08_cnn_normalized_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 9. VISUALIZE THE WORST PERFORMING CATEGORIES
# ============================================================

worst_classes = class_results.head(15).sort_values(
    "f1_score",
    ascending=True
)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=worst_classes,
    x="f1_score",
    y="category"
)

plt.title(
    "15 Lowest Performing Sound Categories",
    fontsize=15
)

plt.xlabel("F1-score")
plt.ylabel("Sound Category")

plt.xlim(0, 1)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "09_cnn_worst_categories.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. VISUALIZE THE MOST FREQUENT CONFUSIONS
# ============================================================

top_pairs = confusion_pairs.head(15).copy()

if len(top_pairs) > 0:

    top_pairs["comparison"] = (
        top_pairs["actual_class"]
        + " → "
        + top_pairs["predicted_class"]
    )

    plt.figure(figsize=(12, 8))

    sns.barplot(
        data=top_pairs,
        x="number_of_errors",
        y="comparison"
    )

    plt.title(
        "Most Frequent CNN Classification Errors",
        fontsize=15
    )

    plt.xlabel("Number of Incorrect Predictions")
    plt.ylabel("Actual → Predicted")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "10_cnn_most_frequent_errors.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("ERROR ANALYSIS SUMMARY")
print("=" * 65)

print("Overall test accuracy:", round(accuracy, 4))

print(
    "Number of correctly classified recordings:",
    int((y_true == y_pred).sum())
)

print(
    "Number of incorrectly classified recordings:",
    len(error_df)
)

print("\nFive lowest F1 categories:")

print(
    class_results[
        ["category", "f1_score", "support"]
    ].head(5).to_string(index=False)
)

print("\nFive most frequent confusion pairs:")

if len(confusion_pairs) > 0:

    print(
        confusion_pairs.head(5).to_string(
            index=False
        )
    )

print("\nFILES SAVED SUCCESSFULLY")
print("-" * 65)

print("cnn_per_class_error_analysis.csv")
print("cnn_most_confused_pairs.csv")
print("cnn_incorrect_predictions.csv")
print("08_cnn_normalized_confusion_matrix.png")
print("09_cnn_worst_categories.png")
print("10_cnn_most_frequent_errors.png")

print("\nPHASE 6 COMPLETED")
