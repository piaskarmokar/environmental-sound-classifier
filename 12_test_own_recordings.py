
# ============================================================
# PHASE 9: TEST CNN ON OUR OWN AUDIO RECORDINGS
# Dataset: ESC-50
# Model: Original trained CNN
# ============================================================

# IMPORTANT: Set environment variables BEFORE importing
# NumPy, Librosa, PyTorch or other scientific libraries.

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Limit OpenMP threads to reduce runtime conflicts.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

RECORDINGS_DIR = PROJECT_DIR / "own_recordings"
RESULTS_DIR = PROJECT_DIR / "results"

TRAIN_FILE = RESULTS_DIR / "train_split.csv"
MODEL_FILE = RESULTS_DIR / "best_cnn_model.pth"

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_FRAMES = 216

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Reduce unnecessary warning messages.
warnings.filterwarnings("ignore")

print("=" * 65)
print("TESTING CNN ON OWN AUDIO RECORDINGS")
print("=" * 65)

print("Device:", DEVICE)


# ============================================================
# 2. CHECK REQUIRED FILES AND FOLDERS
# ============================================================

if not RECORDINGS_DIR.exists():
    raise FileNotFoundError(
        f"Recording folder not found: {RECORDINGS_DIR}"
    )

if not RESULTS_DIR.exists():
    raise FileNotFoundError(
        f"Results folder not found: {RESULTS_DIR}"
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"CNN checkpoint not found: {MODEL_FILE}"
    )

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training split not found: {TRAIN_FILE}"
    )


# ============================================================
# 3. FIND WAV FILES (WITHOUT DUPLICATES)
# ============================================================

audio_files = sorted(
    file
    for file in RECORDINGS_DIR.iterdir()
    if file.is_file() and file.suffix.lower() == ".wav"
)

if not audio_files:
    raise FileNotFoundError(
        "No WAV files found in own_recordings folder."
    )

print("\nRecordings found:")

for file in audio_files:
    print("-", file.name)

print("\nTotal recordings:", len(audio_files))


# ============================================================
# 4. LOAD CLASS NAMES
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)

class_mapping = (
    train_df[["target", "category"]]
    .drop_duplicates()
    .sort_values("target")
)

class_names = class_mapping["category"].tolist()

NUM_CLASSES = len(class_names)

if NUM_CLASSES != 50:
    raise ValueError(
        f"Expected 50 ESC-50 classes, found {NUM_CLASSES}"
    )

print("\nNumber of classes:", NUM_CLASSES)


# ============================================================
# 5. DEFINE ORIGINAL CNN ARCHITECTURE
# ============================================================

class EnvironmentalSoundCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1, 32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128, 128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 256),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# 6. LOAD THE TRAINED CNN
# ============================================================

model = EnvironmentalSoundCNN(
    num_classes=NUM_CLASSES
).to(DEVICE)

try:

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True
    )

except TypeError:

    # Compatibility with older PyTorch versions.
    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )

# Support either a direct state dictionary or a checkpoint
# containing a state_dict.
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

model.load_state_dict(checkpoint)

model.eval()

print("\nOriginal CNN model loaded successfully.")


# ============================================================
# 7. AUDIO PREPROCESSING
# ============================================================

def extract_mel_spectrogram(filepath):

    # Load audio as mono at 22,050 Hz.
    audio, sr = librosa.load(
        filepath,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    # Create Mel spectrogram.
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    # Convert power to decibels.
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
        top_db=80
    )

    # Normalize to the same approximate range
    # used by the original training pipeline.
    mel_db = (mel_db + 80) / 80

    # Pad or truncate to 216 frames.
    if mel_db.shape[1] < TARGET_FRAMES:

        padding = TARGET_FRAMES - mel_db.shape[1]

        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, padding)),
            mode="constant"
        )

    else:

        mel_db = mel_db[:, :TARGET_FRAMES]

    return mel_db.astype(np.float32)


# ============================================================
# 8. PREDICT EACH RECORDING
# ============================================================

results = []

print("\n" + "=" * 65)
print("PREDICTION RESULTS")
print("=" * 65)

for audio_file in audio_files:

    print("\n" + "-" * 55)
    print("Recording:", audio_file.name)

    try:

        # Extract Mel spectrogram.
        mel_feature = extract_mel_spectrogram(audio_file)

        # Expected CNN shape:
        # (batch, channel, mel bands, time frames)
        X = torch.tensor(
            mel_feature,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        X = X.to(DEVICE)

        # Predict.
        with torch.no_grad():

            outputs = model(X)

            probabilities = F.softmax(
                outputs,
                dim=1
            )

            top_probabilities, top_indices = torch.topk(
                probabilities,
                k=5,
                dim=1
            )

        top_probabilities = (
            top_probabilities[0].cpu().numpy()
        )

        top_indices = (
            top_indices[0].cpu().numpy()
        )

        predicted_class = class_names[
            int(top_indices[0])
        ]

        confidence = float(top_probabilities[0])

        # Infer expected category from filename.
        # Examples: Dog.wav -> dog, Rain.wav -> rain.
        expected_label = audio_file.stem.lower().strip()

        # Compare only if the filename matches an ESC-50 class.
        if expected_label in class_names:

            correct = predicted_class == expected_label

        else:

            correct = None

        print("\nPredicted category:", predicted_class)

        print(
            "Top prediction confidence:",
            f"{confidence:.2%}"
        )

        print("\nTop 5 predictions:")

        for rank, (idx, prob) in enumerate(
            zip(top_indices, top_probabilities),
            start=1
        ):

            category = class_names[int(idx)]

            print(
                f"{rank}. {category}: {prob:.2%}"
            )

        if correct is not None:

            print(
                "\nMatches expected filename category:",
                "YES" if correct else "NO"
            )

        else:

            print(
                "\nExpected category not found in ESC-50 labels."
            )

        results.append({

            "filename": audio_file.name,
            "expected_category": expected_label,
            "predicted_category": predicted_class,
            "confidence": confidence,
            "correct": correct

        })

    except Exception as error:

        print("Could not process this recording:", error)

        results.append({

            "filename": audio_file.name,
            "expected_category": audio_file.stem.lower(),
            "predicted_category": "ERROR",
            "confidence": np.nan,
            "correct": None

        })


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_file = (
    RESULTS_DIR / "own_recording_predictions.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("OWN RECORDING EVALUATION SUMMARY")
print("=" * 65)

print(
    results_df.to_string(index=False)
)

successful = results_df[
    results_df["predicted_category"] != "ERROR"
]

print("\nSuccessfully processed recordings:", len(successful))

print("Total recordings:", len(results_df))

print("\nResults saved to:")
print(output_file)

print("\n" + "=" * 65)
print("PHASE 9 COMPLETED")
print("=" * 65)
