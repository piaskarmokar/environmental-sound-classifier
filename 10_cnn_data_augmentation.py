
# ============================================================
# PHASE 8: CNN WITH DATA AUGMENTATION
# Dataset: ESC-50
# Input: Mel Spectrograms
# Model: Environmental Sound CNN
# ============================================================

from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

TRAIN_FILE = RESULTS_DIR / "train_split.csv"
VAL_FILE = RESULTS_DIR / "validation_split.csv"
TEST_FILE = RESULTS_DIR / "test_split.csv"

CACHE_FILE = RESULTS_DIR / "mel_spectrogram_features.npz"

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 7

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("CNN WITH DATA AUGMENTATION")
print("=" * 65)

print("Device:", DEVICE)


# ============================================================
# 2. LOAD DATA AND FEATURES
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

class_mapping = (
    train_df[["target", "category"]]
    .drop_duplicates()
    .sort_values("target")
)

class_names = class_mapping["category"].tolist()

NUM_CLASSES = len(class_names)

print("\nDataset sizes:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

print("Number of classes:", NUM_CLASSES)

if not CACHE_FILE.exists():
    raise FileNotFoundError(
        "Mel spectrogram cache not found. "
        "Run 07_cnn_mel_spectrogram.py first."
    )

cached = np.load(
    CACHE_FILE,
    allow_pickle=False
)

X_all = cached["X"]
cached_filenames = cached["filenames"].tolist()

all_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

all_df = all_df.drop_duplicates(
    subset="filename"
).reset_index(drop=True)

file_list = all_df["filename"].tolist()

if cached_filenames != file_list:
    raise ValueError(
        "Cached feature filenames do not match "
        "the current data splits."
    )

filename_to_index = {
    filename: i
    for i, filename in enumerate(file_list)
}


def prepare_split(df):

    indices = [
        filename_to_index[filename]
        for filename in df["filename"]
    ]

    X = X_all[indices]

    y = df["target"].to_numpy(
        dtype=np.int64
    )

    X = X[:, np.newaxis, :, :]

    return X, y


X_train, y_train = prepare_split(train_df)
X_val, y_val = prepare_split(val_df)
X_test, y_test = prepare_split(test_df)

print("\nFeature dimensions:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ============================================================
# 3. DATA AUGMENTATION
# ============================================================

class SoundDataset(Dataset):

    def __init__(self, X, y, augment=False):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

        self.augment = augment

    def __len__(self):

        return len(self.y)

    def apply_augmentation(self, x):

        # x shape: (1, 128, 216)

        # A. SMALL RANDOM TIME SHIFT

        if random.random() < 0.5:

            shift = random.randint(-12, 12)

            x = torch.roll(
                x,
                shifts=shift,
                dims=2
            )

        # B. ADD SMALL GAUSSIAN NOISE

        if random.random() < 0.5:

            noise = torch.randn_like(x) * 0.015

            x = x + noise

        # C. FREQUENCY MASKING

        if random.random() < 0.7:

            freq_width = random.randint(1, 12)

            freq_start = random.randint(
                0,
                x.shape[1] - freq_width
            )

            x[
                :,
                freq_start:freq_start + freq_width,
                :
            ] = 0

        # D. TIME MASKING

        if random.random() < 0.7:

            time_width = random.randint(1, 20)

            time_start = random.randint(
                0,
                x.shape[2] - time_width
            )

            x[
                :,
                :,
                time_start:time_start + time_width
            ] = 0

        return x.clamp(0, 1)

    def __getitem__(self, index):

        x = self.X[index].clone()

        y = self.y[index]

        if self.augment:

            x = self.apply_augmentation(x)

        return x, y


# Augmentation is enabled ONLY for training data.

train_dataset = SoundDataset(
    X_train,
    y_train,
    augment=True
)

val_dataset = SoundDataset(
    X_val,
    y_val,
    augment=False
)

test_dataset = SoundDataset(
    X_test,
    y_test,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nData augmentation:")
print("Training: ENABLED")
print("Validation: DISABLED")
print("Testing: DISABLED")


# ============================================================
# 4. CNN ARCHITECTURE
# Same architecture as the original CNN
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

            nn.Linear(
                128 * 4 * 4,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = EnvironmentalSoundCNN(
    NUM_CLASSES
).to(DEVICE)

print("\nCNN model created.")


# ============================================================
# 5. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# 6. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0

    for X, y in train_loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * X.size(0)
        )

    return total_loss / len(train_dataset)


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(X)

            loss = criterion(outputs, y)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total_loss += (
                loss.item() * X.size(0)
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
        all_labels,
        all_predictions
    )


# ============================================================
# 8. TRAINING LOOP
# ============================================================

print("\nSTARTING AUGMENTED CNN TRAINING")
print("=" * 65)

best_val_f1 = -1
best_epoch = 0
epochs_without_improvement = 0

history = []

model_file = (
    RESULTS_DIR / "best_augmented_cnn_model.pth"
)

for epoch in range(EPOCHS):

    train_loss = train_one_epoch()

    (
        val_loss,
        val_accuracy,
        val_f1,
        _,
        _
    ) = evaluate(val_loader)

    scheduler.step(val_f1)

    history.append({

        "epoch": epoch + 1,

        "train_loss": train_loss,

        "validation_loss": val_loss,

        "validation_accuracy": val_accuracy,

        "validation_macro_f1": val_f1

    })

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f} | "
        f"Val Macro F1: {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            model_file
        )

        epochs_without_improvement = 0

        print("  Best augmented model saved.")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping activated.")

        break


# ============================================================
# 9. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    RESULTS_DIR / "augmented_cnn_training_history.csv",
    index=False
)

plt.figure(figsize=(9, 5))

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Training Loss"
)

plt.plot(
    history_df["epoch"],
    history_df["validation_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Augmented CNN Training and Validation Loss"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "11_augmented_cnn_training_loss.png",
    dpi=300
)

plt.close()


# ============================================================
# 10. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("FINAL AUGMENTED CNN TEST RESULTS")
print("=" * 65)

model.load_state_dict(
    torch.load(
        model_file,
        map_location=DEVICE
    )
)

(
    test_loss,
    test_accuracy,
    test_macro_f1,
    all_labels,
    all_predictions
) = evaluate(test_loader)

print("Best epoch:", best_epoch)

print(
    "Best validation Macro F1:",
    round(best_val_f1, 4)
)

print("Test loss:", round(test_loss, 4))

print(
    "Test accuracy:",
    round(test_accuracy, 4)
)

print(
    "Test Macro F1:",
    round(test_macro_f1, 4)
)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

plt.figure(figsize=(18, 15))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title(
    "Augmented CNN Confusion Matrix"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "12_augmented_cnn_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "filename": test_df["filename"].tolist(),

    "actual": [
        class_names[i] for i in all_labels
    ],

    "predicted": [
        class_names[i] for i in all_predictions
    ]

})

prediction_df.to_csv(
    RESULTS_DIR / "augmented_cnn_test_predictions.csv",
    index=False
)


# ============================================================
# 14. SAVE EXPERIMENT RESULTS
# ============================================================

results_df = pd.DataFrame([{

    "model": "CNN with Data Augmentation",

    "best_epoch": best_epoch,

    "validation_macro_f1": best_val_f1,

    "test_loss": test_loss,

    "test_accuracy": test_accuracy,

    "test_macro_f1": test_macro_f1

}])

results_df.to_csv(
    RESULTS_DIR / "augmented_cnn_results.csv",
    index=False
)


# ============================================================
# 15. COMPARE ORIGINAL AND AUGMENTED CNN
# ============================================================

original_accuracy = 0.5995
original_macro_f1 = 0.5737

comparison_df = pd.DataFrame({

    "model": [
        "Original CNN",
        "CNN with Data Augmentation"
    ],

    "test_accuracy": [
        original_accuracy,
        test_accuracy
    ],

    "test_macro_f1": [
        original_macro_f1,
        test_macro_f1
    ]

})

comparison_df.to_csv(
    RESULTS_DIR / "cnn_augmentation_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")

print(
    comparison_df.to_string(
        index=False,
        formatters={
            "test_accuracy": "{:.4f}".format,
            "test_macro_f1": "{:.4f}".format
        }
    )
)

print("\nAccuracy improvement (percentage points):")

print(
    round(
        (test_accuracy - original_accuracy) * 100,
        2
    )
)

print("\nMacro F1 improvement (percentage points):")

print(
    round(
        (test_macro_f1 - original_macro_f1) * 100,
        2
    )
)


# ============================================================
# 16. COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("PHASE 8 COMPLETED")
print("=" * 65)

print("Best augmented model:", model_file)

print("Training history saved.")
print("Test predictions saved.")
print("Confusion matrix saved.")
print("Model comparison saved.")

print("\nProcess completed successfully.")
