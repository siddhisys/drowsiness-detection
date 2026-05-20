# 🚗 Driver Drowsiness Detection

A real-time driver drowsiness detection system built using PyTorch and OpenCV, deployed as a Streamlit web application.

---

## Overview

This project implements a binary image classification system that detects driver drowsiness from a live webcam feed or uploaded dashcam footage. Three deep learning models of increasing complexity are trained and compared:

| Model | Architecture | Val Accuracy | AUC-ROC |
|---|---|---|---|
| Baseline CNN | 3-block CNN with Dropout | 100% | 1.0 |
| Advanced CNN | 4-block CNN with BatchNorm + GAP | 100% | 1.0 |
| MobileNetV2 | Transfer learning (ImageNet) | 99.88% | 1.0 |

---

## Features

- 🎥 **Live webcam inference** — real-time frame-by-frame classification
- 📁 **Video upload mode** — analyse pre-recorded dashcam footage
- 🟩 **Face detection** — OpenCV Haar Cascade bounding box around detected face
- 👁️ **Eye detection** — yellow circle markers on detected eyes
- 🔴 **Continuous audio alarm** — fires every 1.5 seconds while drowsiness is detected
- 🔄 **Model switching** — swap between all 3 trained models from the sidebar
- 📊 **Live confidence score** — real-time probability display with progress bar

---

## Project Structure

```
drowsiness-detection/
│
├── dataset/
│   ├── videos/
│   │   ├── alert/          ← alert video recordings
│   │   └── drowsy/         ← drowsy + sleepy video recordings
│   └── frames/
│       ├── alert/          ← extracted face frames (generated)
│       └── drowsy/         ← extracted face frames (generated)
│
├── saved_models/           ← trained model weights (.pt files)
│   ├── baseline_cnn_best.pt
│   ├── advanced_cnn_best.pt
│   └── mobilenet_v2_best.pt
│
├── evaluation_plots/       ← training curves, confusion matrices, ROC curves
│
├── drowsiness_pytorch_final.ipynb  ← training notebook
├── app.py                          ← Streamlit application
├── requirements.txt
└── README.md
```

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/siddhisys/drowsiness-detection.git
cd drowsiness-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your videos

Place your recorded videos in:
- `dataset/videos/alert/` — alert state recordings
- `dataset/videos/drowsy/` — drowsy and sleepy recordings combined

### 4. Run the training notebook

Open and run `drowsiness_pytorch_final.ipynb` top to bottom. This will:
- Extract frames from your videos
- Train all 3 models
- Save weights to `saved_models/`
- Generate evaluation plots

### 5. Launch the app

```bash
python -m streamlit run app.py
```

---

## Model Architecture

### Baseline CNN
A simple 3-block reference model using Conv2D, MaxPooling, and Dropout only.
```
[Conv→ReLU→MaxPool→Dropout] × 3 → Flatten → Dense(128) → Output
```

### Advanced CNN
A deeper 4-block model with BatchNorm, L2 regularisation, and Global Average Pooling.
```
[Conv→BN→ReLU→Conv→BN→ReLU→MaxPool→Dropout] × 4 → GAP → Dense(256) → Dense(128) → Output
```

### MobileNetV2 (Transfer Learning)
Pre-trained ImageNet backbone with a custom classification head, trained in two phases:
- **Phase 1** — backbone frozen, head only (LR = 1e-3)
- **Phase 2** — top 6 backbone layers unfrozen, fine-tuned (LR = 1e-5)

---

## Training Details

| Setting | Value |
|---|---|
| Input size | 96 × 96 grayscale |
| Batch size | 32 |
| Max epochs | 25 |
| Optimiser | Adam |
| Loss function | BCEWithLogitsLoss |
| Early stopping patience | 10 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5) |
| Validation split | 20% |
| Seed | 42 |

---

## Requirements

```
torch
torchvision
opencv-python
streamlit
matplotlib
seaborn
scikit-learn
Pillow
numpy<2
```

---

## Usage — App

1. Run `python -m streamlit run app.py`
2. Select a model from the sidebar
3. Choose **Webcam** or **Upload video**
4. Click **Start Camera** or **Analyse Video**
5. Click **Stop** to end the session

The app will draw a bounding box around the detected face, mark the eyes with yellow circles, and display the prediction label and confidence score in the stats panel. An audio alarm fires continuously while drowsiness is detected.

---

## Known Limitations

- Models were trained on a single subject in a limited number of environments. Performance may vary under significantly different lighting conditions, camera angles, or with different subjects.
- Real-time performance depends on hardware — CPU inference may result in lower frame rates with larger models.
- Streamlit's synchronous execution model can cause minor UI lag during inference.

---

## Future Work

- Expand dataset to include multiple subjects and diverse environments
- Integrate MediaPipe facial landmarks for EAR/MAR-based detection
- Add temporal modelling via LSTM for progressive drowsiness detection
- Extend to multi-class detection (Alert, Yawning, Microsleep, Distracted)
- Optimise for edge deployment via model quantisation and pruning

---

## Author

**Siddhi Jain**  
London Metropolitan University — CT6057NI Computer Vision  
---

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Sandler et al., 2018
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)
