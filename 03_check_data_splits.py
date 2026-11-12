
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
META_FILE = PROJECT_DIR / "meta" / "esc50.csv"

df = pd.read_csv(META_FILE)

print("=" * 55)
print("EXPERIMENTAL DATA SPLIT VERIFICATION")
print("=" * 55)

# 1. Check whether original source files
# appear in multiple folds

source_fold_counts = (
    df.groupby("src_file")["fold"].nunique()
)

violations = source_fold_counts[
    source_fold_counts > 1
]

print("\nUnique original source files:",
      df["src_file"].nunique())

print("Sources appearing in multiple folds:",
      len(violations))

if len(violations) == 0:
    print("PASS: No source-file overlap between folds.")
else:
    print("WARNING: Source-file overlap detected.")
    print(violations)

# 2. Define experimental splits

train_df = df[df["fold"].isin([1, 2, 3])]
val_df = df[df["fold"] == 4]
test_df = df[df["fold"] == 5]

print("\nSplit sizes:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

# 3. Check class distribution

print("\nClass distribution per split:")

for name, subset in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    print(f"\n{name}:")
    print(subset["category"].value_counts().sort_index())

# 4. Check source overlap between splits

train_sources = set(train_df["src_file"])
val_sources = set(val_df["src_file"])
test_sources = set(test_df["src_file"])

print("\nCross-split source overlap:")

print("Train vs validation:",
      len(train_sources & val_sources))

print("Train vs test:",
      len(train_sources & test_sources))

print("Validation vs test:",
      len(val_sources & test_sources))

print("\nSPLIT VERIFICATION COMPLETED")
