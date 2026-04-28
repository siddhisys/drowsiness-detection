"""
app.py — Driver Drowsiness Detection (PyTorch)
Run: streamlit run app.py
"""

import time
import tempfile
import os
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models as tv_models

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
)

IMG_SIZE  = 96
SAVED_DIR = Path(__file__).parent / "saved_models"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FILES = {
    "Baseline CNN": "baseline_cnn_best.pt",
    "Advanced CNN": "advanced_cnn_best.pt",
    "MobileNetV2":  "mobilenet_v2_best.pt",
}

# ── Model definitions ─────────────────────────────────────────────────────────

class BaselineCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel//2, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class AdvancedCNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.block1 = nn.Sequential(ConvBnRelu(in_channels,32), ConvBnRelu(32,32),  nn.MaxPool2d(2), nn.Dropout2d(0.25))
        self.block2 = nn.Sequential(ConvBnRelu(32,64),          ConvBnRelu(64,64),  nn.MaxPool2d(2), nn.Dropout2d(0.25))
        self.block3 = nn.Sequential(ConvBnRelu(64,128),         ConvBnRelu(128,128),nn.MaxPool2d(2), nn.Dropout2d(0.3))
        self.block4 = nn.Sequential(ConvBnRelu(128,256),        ConvBnRelu(256,256))
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128,1),
        )
    def forward(self, x):
        for blk in [self.block1, self.block2, self.block3, self.block4]:
            x = blk(x)
        return self.head(self.gap(x))


class MobileNetV2Drowsiness(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = tv_models.mobilenet_v2(weights=None)
        self.features = backbone.features
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280,256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256,128),  nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128,1),
        )
    def forward(self, x):
        x = x.repeat(1,3,1,1)
        return self.head(self.gap(self.features(x)))


MODEL_CLASSES = {
    "Baseline CNN": BaselineCNN,
    "Advanced CNN": AdvancedCNN,
    "MobileNetV2":  MobileNetV2Drowsiness,
}

# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_name):
    path = SAVED_DIR / MODEL_FILES[model_name]
    if not path.exists():
        return None
    model = MODEL_CLASSES[model_name]()
    model.load_state_dict(torch.load(str(path), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ── Inference ─────────────────────────────────────────────────────────────────

def preprocess_frame(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    tensor  = torch.from_numpy(resized).float() / 255.0
    tensor  = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict(model, frame):
    prob = torch.sigmoid(model(preprocess_frame(frame)).squeeze()).item()
    if prob >= 0.5:
        return "DROWSY", prob
    return "ALERT", 1.0 - prob

def annotate_frame(frame, label, confidence):
    out   = frame.copy()
    color = (0, 0, 220) if label == "DROWSY" else (0, 180, 0)
    cv2.rectangle(out, (0, 0), (frame.shape[1], 50), color, -1)
    cv2.putText(out, f"{label}  {confidence*100:.1f}%", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out

# ── Alarm (beeps every 1.5s while drowsy) ────────────────────────────────────

ALARM_JS = """
<script>
function playAlarm() {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    function beep(freq, start, dur) {
        const o = ctx.createOscillator(), g = ctx.createGain();
        o.connect(g); g.connect(ctx.destination);
        o.frequency.value = freq; o.type = 'square';
        g.gain.setValueAtTime(0.3, ctx.currentTime + start);
        g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + dur);
        o.start(ctx.currentTime + start);
        o.stop(ctx.currentTime + start + dur);
    }
    beep(880, 0.00, 0.25);
    beep(660, 0.30, 0.25);
    beep(880, 0.60, 0.25);
    beep(660, 0.90, 0.25);
}
playAlarm();
</script>
"""

def play_alarm():
    st.components.v1.html(ALARM_JS, height=0)

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🚗 Driver Drowsiness Detection")
st.markdown("Real-time detection from webcam or uploaded dashcam footage.")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Model", list(MODEL_FILES.keys()))
    model = load_model(model_choice)
    if model is None:
        st.error(f"No saved model found for **{model_choice}**. Train it first in the notebook.")
    else:
        st.success(f"{model_choice} loaded ✓")
    st.divider()
    source = st.radio("Input source", ["Webcam", "Upload video"])

col_feed, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("Live stats")
    status_box = st.empty()
    conf_box   = st.empty()
    alert_box  = st.empty()

# ── Inference loop ────────────────────────────────────────────────────────────

def run_inference_loop(frame_source):
    if model is None:
        st.warning("Load a model first.")
        return

    frame_placeholder = col_feed.empty()
    sound_placeholder = col_feed.empty()
    stop_btn = col_feed.button("⏹ Stop", key="stop_btn")
    last_alarm_time = 0

    for frame in frame_source:
        if stop_btn:
            break

        label, conf = predict(model, frame)
        annotated   = annotate_frame(frame, label, conf)
        rgb         = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB", use_column_width=True)

        is_drowsy = label == "DROWSY"
        now = time.time()

        status_box.metric("Status", f"{'🔴' if is_drowsy else '🟢'} {label}")
        conf_box.metric("Confidence", f"{conf*100:.1f}%")

        if is_drowsy:
            alert_box.error("⚠️ DROWSY! Please pull over!")
            if now - last_alarm_time >= 1.5:
                with sound_placeholder:
                    play_alarm()
                last_alarm_time = now
        else:
            alert_box.info("Driving normally")


def webcam_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam.")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def video_file_frames(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip = max(1, int(fps / 10))
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_skip == 0:
                yield frame
            idx += 1
    finally:
        cap.release()
        os.unlink(tmp_path)


# ── Launch ────────────────────────────────────────────────────────────────────

if source == "Webcam":
    with col_feed:
        st.subheader("Webcam feed")
        if st.button("▶ Start webcam", key="start_webcam"):
            run_inference_loop(webcam_frames())
else:
    with col_feed:
        st.subheader("Upload dashcam footage")
        uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded and st.button("▶ Analyse video", key="analyse_video"):
            run_inference_loop(video_file_frames(uploaded))