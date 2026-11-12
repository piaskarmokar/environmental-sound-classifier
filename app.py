import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import gradio as gr
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
TRAIN_FILE = RESULTS_DIR / "train_split.csv"
MODEL_FILE = RESULTS_DIR / "best_cnn_model.pth"

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_FRAMES = 216
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnvironmentalSoundCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_classifier():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TRAIN_FILE}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing file: {MODEL_FILE}")

    train_df = pd.read_csv(TRAIN_FILE)
    class_mapping = (train_df[["target", "category"]]
                     .drop_duplicates().sort_values("target"))
    class_names = class_mapping["category"].tolist()
    if len(class_names) != 50:
        raise ValueError(f"Expected 50 classes; found {len(class_names)}.")

    model = EnvironmentalSoundCNN(len(class_names)).to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=True)
    except TypeError:
        checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model, class_names


MODEL, CLASS_NAMES = load_classifier()


def extract_mel_spectrogram(audio_path):
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError("The uploaded audio file is empty.")

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
    mel_db = (mel_db + 80) / 80

    if mel_db.shape[1] < TARGET_FRAMES:
        padding = TARGET_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, padding)), mode="constant")
    else:
        mel_db = mel_db[:, :TARGET_FRAMES]
    return mel_db.astype(np.float32)


def classify_audio(audio_path):
    if audio_path is None:
        raise gr.Error("Please upload a WAV file.")
    if Path(audio_path).suffix.lower() != ".wav":
        raise gr.Error("Please upload a WAV file.")

    try:
        feature = extract_mel_spectrogram(str(audio_path))
        x = torch.from_numpy(feature).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probabilities = F.softmax(MODEL(x), dim=1)[0]
        scores, indices = torch.topk(probabilities, k=5)

        rows = [
            {
                "Sound category": CLASS_NAMES[int(index)],
                "Model score (%)": round(float(score) * 100, 2)
            }
            for index, score in zip(indices.cpu().tolist(), scores.cpu().tolist())
        ]
        best = rows[0]
        summary = (
            f"## Predicted sound: {best['Sound category']}\n"
            f"**Top model score: {best['Model score (%)']:.2f}%**\n\n"
            "Scores are softmax outputs, not calibrated probabilities. "
            "The model selects among the 50 ESC-50 categories."
        )
        return summary, pd.DataFrame(rows)
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"Could not process this recording: {error}")


with gr.Blocks(title="Environmental Sound Classifier") as demo:
    gr.Markdown(
        "# Environmental Sound Classifier\n"
        "### A CNN trained on the ESC-50 environmental sound dataset\n"
        "Upload a WAV recording to classify it into one of 50 sound categories."
    )
    with gr.Row():
        with gr.Column():
            audio_input = gr.File(
                label="Upload a WAV audio file",
                file_types=[".wav"],
                type="filepath"
            )
            predict_button = gr.Button("Classify sound", variant="primary")
            gr.Markdown(
                "Preprocessing: mono audio at 22,050 Hz; 128 Mel bands; "
                "2,048-point FFT; 512 hop length; 216 frames."
            )
        with gr.Column():
            prediction_output = gr.Markdown("Your prediction will appear here.")
            results_output = gr.Dataframe(
                headers=["Sound category", "Model score (%)"],
                label="Top 5 predictions",
                interactive=False
            )

    predict_button.click(
        fn=classify_audio,
        inputs=audio_input,
        outputs=[prediction_output, results_output]
    )
    gr.Markdown(
        "---\nResearch prototype only. Predictions can be wrong, particularly "
        "for recordings that differ from the ESC-50 dataset."
    )


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Loaded {len(CLASS_NAMES)} ESC-50 categories.")
    demo.launch()
