
# ============================================
# ESC-50 PROJECT
# STEP 1: DATASET VERIFICATION
# ============================================

import pandas as pd
import soundfile as sf

from pathlib import Path

# 1. Locate project folders

PROJECT_DIR = Path(__file__).resolve().parent

AUDIO_DIR = PROJECT_DIR / "audio"
META_FILE = PROJECT_DIR / "meta" / "esc50.csv"

print("=" * 50)
print("ESC-50 DATASET VERIFICATION")
print("=" * 50)

# 2. Check folders and metadata

assert AUDIO_DIR.exists(), "Audio folder not found!"
assert META_FILE.exists(), "Metadata file not found!"

print("\nAudio folder found!")
print("Metadata file found!")

# 3. Read metadata

df = pd.read_csv(META_FILE)

print("\nFirst five records:")
print(df.head())

# 4. Dataset dimensions

print("\nDataset dimensions:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# 5. Count audio files

audio_files = list(AUDIO_DIR.glob("*.wav"))

print("\nTotal audio files:", len(audio_files))

# 6. Count categories

print("\nUnique sound categories:",
      df["category"].nunique())

print("\nRecordings per category:")
print(df["category"].value_counts().sort_index())

# 7. Check folds

print("\nRecordings per fold:")
print(df["fold"].value_counts().sort_index())

# 8. Missing values

print("\nMissing values:")
print(df.isnull().sum())

# 9. Duplicate filenames

print("\nDuplicate filenames:",
      df["filename"].duplicated().sum())

# 10. Check missing audio files

missing_audio = [
    filename
    for filename in df["filename"]
    if not (AUDIO_DIR / filename).exists()
]

print("\nMissing audio files:", len(missing_audio))

# 11. Read one audio recording

sample_file = AUDIO_DIR / df.iloc[0]["filename"]

audio, sample_rate = sf.read(sample_file)

print("\nSample audio information:")
print("Filename:", sample_file.name)
print("Sample rate:", sample_rate)
print("Audio shape:", audio.shape)
print("Duration:", len(audio) / sample_rate, "seconds")

# 12. Final summary

print("\n" + "=" * 50)
print("DATASET VERIFICATION SUMMARY")
print("=" * 50)

print("Metadata records:", len(df))
print("Audio files:", len(audio_files))
print("Categories:", df["category"].nunique())
print("Folds:", df["fold"].nunique())

if (
    len(df) == 2000
    and len(audio_files) == 2000
    and df["category"].nunique() == 50
    and df["fold"].nunique() == 5
    and len(missing_audio) == 0
):
    print("\nSUCCESS! Dataset checks passed.")
else:
    print("\nWARNING! Investigate the dataset.")
