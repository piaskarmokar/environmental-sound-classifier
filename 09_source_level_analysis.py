
# ============================================================
# PHASE 7: SOURCE-LEVEL ERROR ANALYSIS
# Dataset: ESC-50
# Model: CNN Mel Spectrogram
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np


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

print("=" * 65)
print("PHASE 7: SOURCE-LEVEL ERROR ANALYSIS")
print("=" * 65)


# ============================================================
# 2. LOAD DATA
# ============================================================

predictions_df = pd.read_csv(PREDICTIONS_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nPredictions:", predictions_df.shape)
print("Test data:", test_df.shape)


# ============================================================
# 3. MERGE PREDICTIONS WITH TEST METADATA
# ============================================================

df = test_df.merge(
    predictions_df[
        ["filename", "actual", "predicted"]
    ],
    on="filename",
    how="left",
    validate="one_to_one"
)

if df["predicted"].isna().any():
    raise ValueError(
        "Some test recordings have no predictions."
    )

if not (df["actual"] == df["category"]).all():
    raise ValueError(
        "Actual labels do not match test metadata."
    )

df["correct"] = (
    df["actual"] == df["predicted"]
)

df["error"] = (
    ~df["correct"]
)

# Use the ESC-50 metadata source ID.
# Also independently extract it from the filename.

df["source_id_from_filename"] = (
    df["filename"]
    .str.split("-")
    .str[1]
)

df["src_file"] = df["src_file"].astype(str)

df["source_id_from_filename"] = (
    df["source_id_from_filename"].astype(str)
)

if not (
    df["src_file"] == df["source_id_from_filename"]
).all():

    raise ValueError(
        "Source IDs extracted from filenames do not "
        "match the ESC-50 metadata."
    )

df["source_id"] = df["src_file"]

print("\nData verification successful.")

print("Total recordings:", len(df))
print("Unique source IDs:", df["source_id"].nunique())


# ============================================================
# 4. SOURCE-LEVEL PERFORMANCE
# ============================================================

source_results = (
    df.groupby("source_id")
    .agg(

        recordings=("filename", "count"),

        categories=(
            "category",
            lambda x: ", ".join(sorted(set(x)))
        ),

        number_of_categories=(
            "category",
            "nunique"
        ),

        correct_predictions=("correct", "sum"),

        incorrect_predictions=("error", "sum")

    )
    .reset_index()
)

source_results["accuracy"] = (
    source_results["correct_predictions"]
    / source_results["recordings"]
)

source_results["error_rate"] = (
    source_results["incorrect_predictions"]
    / source_results["recordings"]
)

source_results = source_results.sort_values(
    ["incorrect_predictions", "recordings"],
    ascending=[False, False]
)

print("\nSOURCE-LEVEL PERFORMANCE")
print("-" * 65)

print(
    source_results.head(25).to_string(
        index=False
    )
)

source_results.to_csv(
    RESULTS_DIR / "source_level_performance.csv",
    index=False
)


# ============================================================
# 5. IDENTIFY SOURCES WITH REPEATED ERRORS
# ============================================================

repeated_errors = source_results[
    source_results["incorrect_predictions"] >= 2
].copy()

print("\nSOURCE IDs WITH TWO OR MORE INCORRECT PREDICTIONS")
print("-" * 65)

if len(repeated_errors) > 0:

    print(
        repeated_errors.to_string(
            index=False
        )
    )

else:

    print("No source IDs have two or more errors.")

repeated_errors.to_csv(
    RESULTS_DIR / "sources_with_repeated_errors.csv",
    index=False
)


# ============================================================
# 6. CATEGORY-LEVEL SOURCE ANALYSIS
# ============================================================

category_results = (
    df.groupby("category")
    .agg(

        recordings=("filename", "count"),

        unique_sources=("source_id", "nunique"),

        correct_predictions=("correct", "sum"),

        incorrect_predictions=("error", "sum"),

        sources_with_errors=(
            "source_id",
            lambda x: x[
                df.loc[x.index, "error"]
            ].nunique()
        )

    )
    .reset_index()
)

category_results["accuracy"] = (
    category_results["correct_predictions"]
    / category_results["recordings"]
)

category_results["error_rate"] = (
    category_results["incorrect_predictions"]
    / category_results["recordings"]
)

category_results["errors_per_source"] = (
    category_results["incorrect_predictions"]
    / category_results["unique_sources"]
)

category_results = category_results.sort_values(
    "error_rate",
    ascending=False
)

print("\nCATEGORY-LEVEL SOURCE ANALYSIS")
print("-" * 65)

print(
    category_results.to_string(
        index=False
    )
)

category_results.to_csv(
    RESULTS_DIR / "category_source_analysis.csv",
    index=False
)


# ============================================================
# 7. INVESTIGATE THE MOST DIFFICULT SOURCES
# ============================================================

print("\nDETAILED RECORDINGS FROM THE WORST SOURCES")
print("-" * 65)

worst_source_ids = (
    repeated_errors["source_id"]
    .head(10)
    .tolist()
)

if len(worst_source_ids) > 0:

    worst_source_recordings = df[
        df["source_id"].isin(worst_source_ids)
    ][
        [
            "source_id",
            "filename",
            "category",
            "actual",
            "predicted",
            "correct"
        ]
    ].sort_values(
        ["source_id", "filename"]
    )

    print(
        worst_source_recordings.to_string(
            index=False
        )
    )

    worst_source_recordings.to_csv(
        RESULTS_DIR / "worst_source_recordings.csv",
        index=False
    )

else:

    print("No repeated-error source groups to investigate.")


# ============================================================
# 8. MEASURE ERROR CONCENTRATION
# ============================================================

total_errors = int(df["error"].sum())

top_10_sources = source_results.head(10)

errors_in_top_10 = int(
    top_10_sources["incorrect_predictions"].sum()
)

if total_errors > 0:

    top_10_error_percentage = (
        errors_in_top_10 / total_errors
    ) * 100

else:

    top_10_error_percentage = 0

sources_with_errors = int(
    (source_results["incorrect_predictions"] > 0).sum()
)

sources_with_repeated_errors = int(
    (source_results["incorrect_predictions"] >= 2).sum()
)

print("\nERROR CONCENTRATION")
print("-" * 65)

print("Total incorrect predictions:", total_errors)

print(
    "Sources with at least one error:",
    sources_with_errors
)

print(
    "Sources with two or more errors:",
    sources_with_repeated_errors
)

print(
    "Errors from the 10 highest-error sources:",
    errors_in_top_10
)

print(
    "Percentage of errors from these 10 sources:",
    round(top_10_error_percentage, 2),
    "%"
)


# ============================================================
# 9. RECORDINGS PER SOURCE
# ============================================================

print("\nSOURCE RECORDING DISTRIBUTION")
print("-" * 65)

print(
    source_results["recordings"]
    .value_counts()
    .sort_index()
    .rename_axis("recordings_per_source")
    .reset_index(name="number_of_sources")
    .to_string(index=False)
)


# ============================================================
# 10. SAVE FINAL SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "total_test_recordings": len(df),

    "unique_source_ids": df["source_id"].nunique(),

    "total_errors": total_errors,

    "sources_with_errors": sources_with_errors,

    "sources_with_repeated_errors": (
        sources_with_repeated_errors
    ),

    "top_10_source_errors": errors_in_top_10,

    "top_10_error_percentage": (
        top_10_error_percentage
    )

}])

summary.to_csv(
    RESULTS_DIR / "source_analysis_summary.csv",
    index=False
)


# ============================================================
# 11. COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("PHASE 7 COMPLETED")
print("=" * 65)

print("\nFiles saved:")

print("1. source_level_performance.csv")
print("2. sources_with_repeated_errors.csv")
print("3. category_source_analysis.csv")
print("4. worst_source_recordings.csv")
print("5. source_analysis_summary.csv")

print("\nProcess completed successfully.")
