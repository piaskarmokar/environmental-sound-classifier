
# ============================================================
# ESC-50 DEEP LEARNING PROJECT
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

from pathlib import Path

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

AUDIO_DIR = PROJECT_DIR / "audio"
META_FILE = PROJECT_DIR / "meta" / "esc50.csv"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Read metadata
df = pd.read_csv(META_FILE)

print("=" * 60)
print("ESC-50 EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# ============================================================
# 2. DATASET INFORMATION
# ============================================================

print("\n2. DATASET INFORMATION")

print("\nDataset structure:")
df.info()

print("\nFirst five observations:")
print(df.head())

print("\nDataset dimensions:", df.shape)

print("\nNumber of categories:", df["category"].nunique())

print("\nCategory distribution:")
print(df["category"].value_counts().sort_index())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate filenames:",
      df["filename"].duplicated().sum())


# ============================================================
# 3. CATEGORY DISTRIBUTION
# ============================================================

print("\n3. CATEGORY DISTRIBUTION")

category_counts = (
    df["category"]
    .value_counts()
    .sort_values()
)

fig, ax = plt.subplots(figsize=(12, 14))

category_counts.plot(
    kind="barh",
    ax=ax,
    edgecolor="black",
    linewidth=0.5
)

ax.set_title(
    "Distribution of ESC-50 Sound Categories",
    fontsize=15,
    pad=15
)

ax.set_xlabel("Number of Recordings")
ax.set_ylabel("Sound Category")

ax.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "01_category_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 4. AUDIO METADATA ANALYSIS
# ============================================================

print("\n4. AUDIO METADATA ANALYSIS")

audio_durations = []
audio_amplitudes = []

for filename in df["filename"]:

    file_path = AUDIO_DIR / filename

    audio, sr = sf.read(file_path)

    duration = len(audio) / sr

    amplitude = np.max(np.abs(audio))

    audio_durations.append(duration)
    audio_amplitudes.append(amplitude)

df["duration"] = audio_durations
df["max_amplitude"] = audio_amplitudes

print("\nAudio duration summary:")
print(df["duration"].describe())

print("\nMaximum amplitude summary:")
print(df["max_amplitude"].describe())

# Save audio metadata summary

df[
    ["filename", "category", "fold",
     "duration", "max_amplitude"]
].to_csv(
    RESULTS_DIR / "audio_metadata_summary.csv",
    index=False
)


# ============================================================
# 5. LOAD SAMPLE AUDIO
# ============================================================

print("\n5. SAMPLE AUDIO ANALYSIS")

sample_row = df[
    df["category"] == "dog"
].iloc[0]

sample_path = AUDIO_DIR / sample_row["filename"]

audio, sr = librosa.load(
    sample_path,
    sr=22050,
    mono=True
)

print("\nSample recording:")
print("Filename:", sample_row["filename"])
print("Category:", sample_row["category"])
print("Sampling rate:", sr)
print("Duration:", len(audio) / sr)
print("Audio samples:", len(audio))


# ============================================================
# 6. WAVEFORM VISUALIZATION
# ============================================================

print("\n6. WAVEFORM VISUALIZATION")

time = np.arange(len(audio)) / sr

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    time,
    audio,
    linewidth=0.6
)

ax.set_title(
    "Waveform of a Dog Sound Recording",
    fontsize=14
)

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude")

ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "02_dog_waveform.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 7. MEL SPECTROGRAM
# ============================================================

print("\n7. MEL SPECTROGRAM ANALYSIS")

mel_spectrogram = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)

mel_db = librosa.power_to_db(
    mel_spectrogram,
    ref=np.max
)

fig, ax = plt.subplots(figsize=(12, 5))

img = librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=512,
    x_axis="time",
    y_axis="mel",
    ax=ax
)

fig.colorbar(
    img,
    ax=ax,
    format="%+2.0f dB"
)

ax.set_title(
    "Mel Spectrogram of a Dog Sound Recording",
    fontsize=14
)

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Mel Frequency")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "03_dog_mel_spectrogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 8. COMPARE DIFFERENT SOUND CATEGORIES
# ============================================================

print("\n8. WAVEFORM COMPARISON")

categories = ["dog", "rain", "airplane"]

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 10),
    constrained_layout=True
)

for ax, category in zip(axes, categories):

    row = df[
        df["category"] == category
    ].iloc[0]

    file_path = AUDIO_DIR / row["filename"]

    sound, sample_rate = librosa.load(
        file_path,
        sr=22050,
        mono=True
    )

    sound_time = np.arange(len(sound)) / sample_rate

    ax.plot(
        sound_time,
        sound,
        linewidth=0.6
    )

    ax.set_title(
        f"Waveform: {category.title()}",
        fontsize=12
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")

    ax.grid(alpha=0.25)

fig.savefig(
    RESULTS_DIR / "04_waveform_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. AUDIO DURATION DISTRIBUTION
# ============================================================

print("\n9. AUDIO DURATION DISTRIBUTION")

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    df["duration"],
    bins=20,
    edgecolor="black"
)

ax.set_title("Distribution of Audio Recording Durations")

ax.set_xlabel("Duration (seconds)")
ax.set_ylabel("Number of Recordings")

ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "05_audio_duration_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Total recordings:", len(df))

print("Total categories:",
      df["category"].nunique())

print("Average duration:",
      round(df["duration"].mean(), 3),
      "seconds")

print("Minimum duration:",
      round(df["duration"].min(), 3),
      "seconds")

print("Maximum duration:",
      round(df["duration"].max(), 3),
      "seconds")

print("\nAll figures and summaries saved in:")
print(RESULTS_DIR)

print("\nEDA PHASE FINISHED!")
