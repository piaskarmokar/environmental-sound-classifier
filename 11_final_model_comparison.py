
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

BASELINE_FILE = RESULTS_DIR / "baseline_results.csv"
CNN_FILE = RESULTS_DIR / "cnn_results.csv"
AUGMENTED_FILE = RESULTS_DIR / "augmented_cnn_results.csv"

OUTPUT_CSV = RESULTS_DIR / "final_model_comparison.csv"
OUTPUT_FIGURE = RESULTS_DIR / "13_final_model_comparison.png"


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

print("=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

required_files = [
    BASELINE_FILE,
    CNN_FILE,
    AUGMENTED_FILE
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

print("\nAll three model result files found.")


# ============================================================
# 3. LOAD RESULTS
# ============================================================

baseline = pd.read_csv(BASELINE_FILE)
cnn = pd.read_csv(CNN_FILE)
augmented = pd.read_csv(AUGMENTED_FILE)


# ============================================================
# 4. STANDARDIZE MODEL RESULTS
# ============================================================

baseline_row = {
    "Model": "MFCC + Logistic Regression",
    "Validation Macro F1": baseline.loc[0, "Validation_Macro_F1"],
    "Test Accuracy": baseline.loc[0, "Test_Accuracy"],
    "Test Macro F1": baseline.loc[0, "Test_Macro_F1"],
    "Test Loss": np.nan
}

cnn_row = {
    "Model": "Original CNN",
    "Validation Macro F1": cnn.loc[0, "validation_macro_f1"],
    "Test Accuracy": cnn.loc[0, "test_accuracy"],
    "Test Macro F1": cnn.loc[0, "test_macro_f1"],
    "Test Loss": cnn.loc[0, "test_loss"]
}

augmented_row = {
    "Model": "CNN with Data Augmentation",
    "Validation Macro F1": augmented.loc[0, "validation_macro_f1"],
    "Test Accuracy": augmented.loc[0, "test_accuracy"],
    "Test Macro F1": augmented.loc[0, "test_macro_f1"],
    "Test Loss": augmented.loc[0, "test_loss"]
}

comparison = pd.DataFrame([
    baseline_row,
    cnn_row,
    augmented_row
])


# ============================================================
# 5. SAVE CONSOLIDATED RESULTS
# ============================================================

comparison.to_csv(OUTPUT_CSV, index=False)

print("\nFINAL MODEL PERFORMANCE")
print("=" * 65)

display_table = comparison.copy()

for column in [
    "Validation Macro F1",
    "Test Accuracy",
    "Test Macro F1"
]:
    display_table[column] = (
        display_table[column] * 100
    ).round(2)

display_table["Test Loss"] = (
    display_table["Test Loss"].round(4)
)

print(display_table.to_string(index=False))


# ============================================================
# 6. CALCULATE PERFORMANCE DIFFERENCES
# ============================================================

original_cnn_accuracy = cnn_row["Test Accuracy"]
baseline_accuracy = baseline_row["Test Accuracy"]
augmented_accuracy = augmented_row["Test Accuracy"]

original_cnn_f1 = cnn_row["Test Macro F1"]
baseline_f1 = baseline_row["Test Macro F1"]
augmented_f1 = augmented_row["Test Macro F1"]

print("\nPERFORMANCE IMPROVEMENTS")
print("=" * 65)

print(
    "Original CNN vs Logistic Regression:"
)

print(
    f"Accuracy improvement: "
    f"{(original_cnn_accuracy - baseline_accuracy) * 100:.2f} "
    f"percentage points"
)

print(
    f"Macro F1 improvement: "
    f"{(original_cnn_f1 - baseline_f1) * 100:.2f} "
    f"percentage points"
)

print(
    "\nOriginal CNN vs Augmented CNN:"
)

print(
    f"Accuracy difference: "
    f"{(original_cnn_accuracy - augmented_accuracy) * 100:.2f} "
    f"percentage points"
)

print(
    f"Macro F1 difference: "
    f"{(original_cnn_f1 - augmented_f1) * 100:.2f} "
    f"percentage points"
)


# ============================================================
# 7. CREATE FINAL COMPARISON FIGURE
# ============================================================

models = [
    "MFCC + Logistic\nRegression",
    "Original CNN",
    "CNN with Data\nAugmentation"
]

accuracy_values = [
    baseline_accuracy * 100,
    original_cnn_accuracy * 100,
    augmented_accuracy * 100
]

f1_values = [
    baseline_f1 * 100,
    original_cnn_f1 * 100,
    augmented_f1 * 100
]

x = np.arange(len(models))
width = 0.34

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(
    x - width / 2,
    accuracy_values,
    width,
    label="Test Accuracy"
)

bars2 = ax.bar(
    x + width / 2,
    f1_values,
    width,
    label="Test Macro F1"
)

# Add values above the bars
ax.bar_label(
    bars1,
    fmt="%.2f%%",
    padding=3,
    fontsize=9
)

ax.bar_label(
    bars2,
    fmt="%.2f%%",
    padding=3,
    fontsize=9
)

ax.set_title(
    "Performance Comparison of Three Classification Models",
    fontsize=14,
    pad=15
)

ax.set_ylabel("Performance (%)", fontsize=11)

ax.set_xlabel("Classification Model", fontsize=11)

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylim(0, 75)

ax.legend()

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("FINAL EVALUATION COMPLETED")
print("=" * 65)

print(f"\nResults CSV: {OUTPUT_CSV}")
print(f"Comparison figure: {OUTPUT_FIGURE}")

print("\nBest model based on test macro F1: Original CNN")

print(
    f"Test Accuracy: {original_cnn_accuracy * 100:.2f}%"
)

print(
    f"Test Macro F1: {original_cnn_f1 * 100:.2f}%"
)

print("\nProcess completed successfully.")
