
import pandas as pd
from pathlib import Path

# ==========================================
# CREATE LEAKAGE-CONTROLLED DATA SPLITS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

META_FILE = PROJECT_DIR / "meta" / "esc50.csv"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(META_FILE)

# 1. Initial official fold assignments

train_df = df[df["fold"].isin([1, 2, 3])].copy()

val_df = df[df["fold"] == 4].copy()

test_df = df[df["fold"] == 5].copy()

# 2. Identify source IDs shared by validation and test

shared_sources = (
    set(val_df["src_file"]) &
    set(test_df["src_file"])
)

print("Shared validation-test sources:", shared_sources)

# 3. Move all validation recordings of shared
# sources into the test set

move_to_test = val_df[
    val_df["src_file"].isin(shared_sources)
].copy()

val_df = val_df[
    ~val_df["src_file"].isin(shared_sources)
].copy()

test_df = pd.concat(
    [test_df, move_to_test],
    ignore_index=True
)

# 4. Check final split sizes

print("\nFINAL SPLIT SIZES")

print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

# 5. Check source overlap

train_sources = set(train_df["src_file"])
val_sources = set(val_df["src_file"])
test_sources = set(test_df["src_file"])

print("\nSOURCE OVERLAP CHECK")

print("Train vs Validation:",
      len(train_sources & val_sources))

print("Train vs Test:",
      len(train_sources & test_sources))

print("Validation vs Test:",
      len(val_sources & test_sources))

# 6. Check class distribution

print("\nCLASS DISTRIBUTION")

for name, subset in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    print(f"\n{name}:")
    print(subset["category"].value_counts().sort_index())

# 7. Save final splits

train_df.to_csv(
    RESULTS_DIR / "train_split.csv",
    index=False
)

val_df.to_csv(
    RESULTS_DIR / "validation_split.csv",
    index=False
)

test_df.to_csv(
    RESULTS_DIR / "test_split.csv",
    index=False
)

# 8. Final verification

assert len(train_df) + len(val_df) + len(test_df) == 2000

assert len(train_sources & val_sources) == 0
assert len(train_sources & test_sources) == 0
assert len(val_sources & test_sources) == 0

assert (
    train_df["filename"].nunique()
    + val_df["filename"].nunique()
    + test_df["filename"].nunique()
    == 2000
)

print("\nSUCCESS!")
print("All recordings retained.")
print("No original source appears across splits.")
print("Final split files saved in results folder.")
