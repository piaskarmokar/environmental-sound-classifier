# environmental-sound-classifier
# Environmental Sound Classification Using Deep Learning

**Project:** Environmental Sound Classification Using Deep Learning: A
Comparison of CNN and Traditional Machine Learning\
**Dataset:** ESC-50\
**Author:** Pias Karmokar

## 1. Project overview

This project investigates whether a Convolutional Neural Network (CNN)
can classify environmental sounds more effectively than a traditional
machine-learning baseline.

The experiments compare:

1.  **MFCC + Logistic Regression:** 40 MFCCs summarised by their mean
    and standard deviation (80 features per recording), followed by
    Logistic Regression.
2.  **CNN + Mel spectrogram:** a convolutional neural network trained on
    Mel-spectrogram representations.
3.  **CNN + spectrogram augmentation:** the same CNN approach with time
    shifting, additive spectrogram noise, frequency masking, and time
    masking during training.

The project also includes per-class error analysis, source-level
analysis, and a small qualitative test using three independently
recorded sounds.

### Research question

> How effectively can a Convolutional Neural Network (CNN) classify
> environmental sounds, and does it outperform a traditional
> machine-learning baseline?

## 2. Dataset

The project uses **ESC-50**, a labelled environmental sound dataset.

-   2,000 audio recordings
-   50 sound classes, with 40 recordings per class
-   Recordings are approximately 5 seconds long
-   Mono audio, originally sampled at 44.1 kHz
-   Five predefined folds
-   Official repository: https://github.com/karolpiczak/ESC-50

ESC-50 is distributed under the **Creative Commons
Attribution-NonCommercial (CC BY-NC)** licence. Check the licence before
reuse, especially for commercial purposes.

**Dataset paper:** Piczak, K. J. (2015). "ESC: Dataset for Environmental
Sound Classification." Proceedings of the 23rd ACM International
Conference on Multimedia.

## 3. Project structure

Run the scripts from the project root directory.

``` text
Final/
├── audio/                         # ESC-50 WAV files (download separately)
├── meta/                          # ESC-50 metadata (download separately)
├── own_recordings/                # Optional independent demo recordings
├── results/                       # Generated splits, CSV results and analysis
├── 01_check_dataset.py
├── 02_exploratory_data_analysis.py
├── 03_check_data_splits.py
├── 04_investigate_source_overlap.py
├── 05_create_final_splits.py
├── 06_mfcc_baseline_model.py
├── 07_cnn_mel_spectrogram.py
├── 08_error_analysis.py
├── 09_source_level_analysis.py
├── 10_cnn_data_augmentation.py
├── 12_test_own_recordings.py
└── README.md
```

The scripts also generate figures, model checkpoints, feature caches,
and intermediate CSV files. These are created during execution and are
not all listed in this abbreviated tree.

## 4. Setup

### Requirements

The project uses Python and the following packages:

-   numpy
-   pandas
-   librosa
-   soundfile
-   scikit-learn
-   matplotlib
-   seaborn
-   torch (PyTorch)

Create and activate a virtual environment (recommended):

``` bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

``` bash
python -m pip install --upgrade pip
python -m pip install numpy pandas librosa soundfile scikit-learn matplotlib seaborn torch
```

If the standard PyTorch installation does not suit your computer, use
the official installation selector:
https://pytorch.org/get-started/locally/.

## 5. Download the dataset

Clone the official ESC-50 repository:

``` bash
git clone https://github.com/karolpiczak/ESC-50.git
```

Copy its `audio` and `meta` directories into this project's root
directory. The expected layout is:

``` text
Final/
├── audio/
│   ├── 1-100032-A-0.wav
│   └── ...
└── meta/
    └── esc50.csv
```

Use the original audio filenames and metadata. If you change them,
update the scripts accordingly.

## 6. Reproduce the experiments

Run scripts from the project root, in this order:

  ------------------------------------------------------------------------------------
  Order                   Script                               Purpose
  ----------------------- ------------------------------------ -----------------------
  1                       `01_check_dataset.py`                Verify audio files,
                                                               metadata, class counts,
                                                               and recording
                                                               properties.

  2                       `02_exploratory_data_analysis.py`    Explore class
                                                               distribution and audio
                                                               characteristics;
                                                               generate waveform and
                                                               spectrogram figures.

  3                       `03_check_data_splits.py`            Inspect the original
                                                               fold structure.

  4                       `04_investigate_source_overlap.py`   Investigate source
                                                               identifiers appearing
                                                               in more than one
                                                               original fold.

  5                       `05_create_final_splits.py`          Create the modified
                                                               source-aware training,
                                                               validation, and test
                                                               partitions.

  6                       `06_mfcc_baseline_model.py`          Extract MFCC summary
                                                               features, select
                                                               Logistic Regression
                                                               regularisation using
                                                               validation data, and
                                                               evaluate the baseline.

  7                       `07_cnn_mel_spectrogram.py`          Prepare Mel
                                                               spectrograms, train the
                                                               CNN, select a
                                                               checkpoint using
                                                               validation performance,
                                                               and evaluate on the
                                                               test set.

  8                       `08_error_analysis.py`               Examine per-class
                                                               performance and
                                                               frequent classification
                                                               errors.

  9                       `09_source_level_analysis.py`        Examine whether errors
                                                               are concentrated in
                                                               particular source
                                                               recordings.

  10                      `10_cnn_data_augmentation.py`        Train and evaluate a
                                                               CNN with
                                                               spectrogram-level
                                                               augmentation.

  11                      `12_test_own_recordings.py`          Test the trained CNN on
                                                               recordings in
                                                               `own_recordings/`
                                                               (optional).
  ------------------------------------------------------------------------------------

Example:

``` bash
python 01_check_dataset.py
```

On Windows, you can also run a script using the full path to your Python
interpreter and the script file.

**Execution order matters:** generate the split CSV files before running
the model scripts. The own-recording script requires the trained CNN
checkpoint and WAV files in `own_recordings/`. If a script reports a
missing file, confirm that earlier steps completed and that the command
was run from the project root.

## 7. Data splitting and evaluation

The original ESC-50 folds were checked for repeated source identifiers.
A small number of source identifiers appeared in more than one original
fold. To reduce potential source-level leakage, the project created a
**modified source-aware split**:

-   **Training:** original folds 1, 2, and 3 --- 1,200 recordings.
-   **Validation:** fold 4 after removing recordings whose source
    identifiers also occur in fold 5 --- 398 recordings.
-   **Test:** original fold 5 plus the removed overlapping-source
    recordings from fold 4 --- 402 recordings.

All 2,000 recordings are retained. The final partitions have no shared
source identifiers according to the project's metadata checks.

This is a project-specific modified split, not the unchanged official
five-fold evaluation protocol. Interpret the reported results in that
context.

Model choices were made using the validation set, while the test set was
reserved for final evaluation. The main metrics are **accuracy** and
**macro F1-score**. Macro F1 gives equal weight to each class.

## 8. Main results

Recorded test-set results:

  Model                                 Test accuracy   Test macro F1
  ----------------------------------- --------------- ---------------
  MFCC + Logistic Regression                   41.79%          40.15%
  CNN on Mel spectrograms                  **59.95%**      **57.37%**
  CNN with spectrogram augmentation            54.73%          51.67%

The original CNN improved over the MFCC + Logistic Regression baseline
by **18.16 percentage points in accuracy** and **17.22 percentage points
in macro F1**. The tested augmentation settings did not improve
performance.

Performance varied across sound classes. The error analysis identified
both well-classified categories and categories with substantial
confusion. Some errors were concentrated in a small number of sources,
but source concentration did not explain all model errors.

## 9. Independent recording demonstration

The optional `own_recordings/` folder contains three independently
recorded WAV files:

``` text
own_recordings/
├── Dog.wav
├── Footsteps.wav
└── Rain.wav
```

The completed test produced these top predictions:

  Recording       Top predicted class     Model score
  --------------- --------------------- -------------
  Dog.wav         cow                          38.44%
  Footsteps.wav   snoring                      46.32%
  Rain.wav        snoring                      71.55%

None of the three recordings was classified as its intended sound. This
is a **small qualitative demonstration**, not an estimate of real-world
accuracy. Three recordings are insufficient for a general performance
claim. Recording conditions, background sounds, and preprocessing
differences may affect predictions.

Before using this as a live demonstration, verify that the
independent-recording script uses the same sample rate, duration
handling, Mel-spectrogram settings, and normalization as the CNN
training pipeline.

## 10. Reproducibility notes and limitations

-   The dataset must be downloaded separately; it is not bundled with
    the project.
-   The split is modified to account for repeated source identifiers. Do
    not describe it as the untouched official ESC-50 fold split.
-   Results may vary with random seeds, software versions, and hardware.
-   The augmentation experiment tested a particular set of spectrogram
    transformations; it does not show that all forms of augmentation are
    ineffective.
-   The three independent recordings are only a qualitative
    demonstration.
-   This 50-class model is a research prototype, not a validated
    real-world sound-monitoring system.
-   ESC-50 is licensed CC BY-NC. Check the licence before using the
    dataset or derived material commercially.

## 11. Suggested citation

Piczak, K. J. (2015). "ESC: Dataset for Environmental Sound
Classification." Proceedings of the 23rd ACM International Conference on
Multimedia.

Dataset repository: https://github.com/karolpiczak/ESC-50
