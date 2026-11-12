
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    PROJECT_DIR / "meta" / "esc50.csv"
)

# Find original sources appearing in multiple folds

source_fold_counts = (
    df.groupby("src_file")["fold"].nunique()
)

overlapping_sources = source_fold_counts[
    source_fold_counts > 1
].index

# Display all recordings belonging to those sources

overlap_df = df[
    df["src_file"].isin(overlapping_sources)
].sort_values(
    ["src_file", "fold"]
)

print("=" * 60)
print("OVERLAPPING SOURCE INVESTIGATION")
print("=" * 60)

print(
    overlap_df[
        ["filename", "src_file", "take",
         "category", "fold"]
    ].to_string(index=False)
)

print("\nSource overlap by fold:")

for source_id in overlapping_sources:

    group = overlap_df[
        overlap_df["src_file"] == source_id
    ]

    print(
        f"\nSource {source_id}: "
        f"Folds {sorted(group['fold'].unique())}"
    )
    print(group[
        ["filename", "category", "fold"]
    ].to_string(index=False))

print("\nInvestigation completed.")
