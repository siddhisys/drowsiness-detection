"""
app.py — Driver Drowsiness Detection
======================================
Real-time drowsiness detection using:
  - OpenCV     : face detection via Haar Cascade + bounding box
  - PyTorch    : trained CNN models for alert/drowsy classification
  - Streamlit  : web-based UI with live webcam feed

Run: python -m streamlit run app.py
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE  = 96
SAVED_DIR = Path(__file__).parent / "saved_models"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES   = ["Alert", "Drowsy"]

MODEL_FILES = {
    "Baseline CNN": "baseline_cnn_best.pt",
    "Advanced CNN": "advanced_cnn_best.pt",
    "MobileNetV2":  "mobilenet_v2_best.pt",
}

# ── OpenCV Haar Cascade face detector ────────────────────────────────────────
# haarcascade_frontalface_default.xml ships with OpenCV — no extra install needed
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# Eye cascade for drawing eye landmarks
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ── Model definitions (must exactly match the training notebook) ──────────────

class BaselineCNN(nn.Module):
    """Simple 3-block CNN reference model."""
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
    """Conv → BatchNorm → ReLU building block."""
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class AdvancedCNN(nn.Module):
    """4 double-conv blocks with BatchNorm + GlobalAveragePooling."""
    def __init__(self, in_channels=1):
        super().__init__()
        self.block1 = nn.Sequential(ConvBnRelu(in_channels,32), ConvBnRelu(32,32),  nn.MaxPool2d(2), nn.Dropout2d(0.25))
        self.block2 = nn.Sequential(ConvBnRelu(32,64),          ConvBnRelu(64,64),  nn.MaxPool2d(2), nn.Dropout2d(0.25))
        self.block3 = nn.Sequential(ConvBnRelu(64,128),         ConvBnRelu(128,128),nn.MaxPool2d(2), nn.Dropout2d(0.30))
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
    """MobileNetV2 backbone with custom head. Grayscale → 3 channels internally."""
    def __init__(self):
        super().__init__()
        backbone      = tv_models.mobilenet_v2(weights=None)
        self.features = backbone.features
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280,256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256,128),  nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128,1),
        )
    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.head(self.gap(self.features(x)))


MODEL_CLASSES = {
    "Baseline CNN": BaselineCNN,
    "Advanced CNN": AdvancedCNN,
    "MobileNetV2":  MobileNetV2Drowsiness,
}

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    """
    Load saved PyTorch model weights from saved_models/.
    Cached so it only loads once per session.
    Returns None if file not found.
    """
    path = SAVED_DIR / MODEL_FILES[model_name]
    if not path.exists():
        return None
    model = MODEL_CLASSES[model_name]()
    model.load_state_dict(torch.load(str(path), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Disable dropout for inference
    return model


# ── Preprocessing & inference ─────────────────────────────────────────────────
def preprocess(frame):
    """
    Convert a BGR face crop to a normalised tensor for inference.
    Must match training preprocessing:
    grayscale → resize 96x96 → normalise to [-1,1] → shape (1,1,96,96)
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    tensor  = torch.from_numpy(resized).float() / 255.0
    tensor  = (tensor - 0.5) / 0.5                     # Normalise to [-1, 1]
    return tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, 96, 96)


@torch.no_grad()
def predict(model, face_crop, threshold=0.5):
    """
    Run inference on a face crop.
    Returns (label, confidence) where label is 'Alert' or 'Drowsy'.
    """
    prob  = torch.sigmoid(model(preprocess(face_crop)).squeeze()).item()
    if prob >= threshold:
        return "Drowsy", prob
    return "Alert", 1.0 - prob


# ── Face detection + annotation ───────────────────────────────────────────────
def process_frame(frame, model):
    """
    Full per-frame pipeline using OpenCV:
    1. Convert to grayscale for Haar Cascade detection
    2. Detect face → draw coloured bounding box
    3. Crop face → run model inference
    4. Detect eyes within face region → draw eye markers
    5. Draw prediction label and confidence on frame

    Returns annotated frame, label string, confidence score.
    """
    out   = frame.copy()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w  = frame.shape[:2]

    label, conf = "No Face", 0.0

    # ── Haar Cascade face detection ───────────────────────────────────────────
    # scaleFactor: how much image size is reduced at each scale
    # minNeighbors: how many neighbours a rectangle needs to be retained
    # minSize: minimum face size to detect
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) > 0:
        # Use the largest detected face (most likely the driver)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, fw, fh = faces[0]

        # Add 20% margin padding around detected face
        margin = int(0.2 * min(fw, fh))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + fw + margin)
        y2 = min(h, y + fh + margin)

        # Crop face and run model inference
        face_crop   = frame[y1:y2, x1:x2]
        if model is not None and face_crop.size > 0:
            label, conf = predict(model, face_crop)

        # Bounding box colour: red = drowsy, green = alert
        box_color = (0, 0, 220) if label == "Drowsy" else (0, 200, 0)

        # Draw face bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

        # Draw label above bounding box
        label_text = f"{label}  {conf*100:.1f}%"
        cv2.putText(out, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)

        # ── Eye detection within face region ──────────────────────────────────
        # Detect eyes only in the top half of the face crop (avoids false positives)
        face_gray    = gray[y:y+fh, x:x+fw]
        top_half     = face_gray[:fh//2, :]  # Eyes are in upper half of face
        eyes = EYE_CASCADE.detectMultiScale(
            top_half,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )

        # Draw circles on detected eye centres
        for (ex, ey, ew, eh) in eyes[:2]:  # Max 2 eyes
            # Convert eye coords back to full frame coords
            eye_cx = x + ex + ew // 2
            eye_cy = y + ey + eh // 2
            cv2.circle(out, (eye_cx, eye_cy), 8,  (0, 255, 255), 2)   # Yellow ring
            cv2.circle(out, (eye_cx, eye_cy), 2,  (0, 255, 255), -1)  # Yellow dot

        # ── Coloured overlay bar at top when drowsy ───────────────────────────
        if label == "Drowsy":
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
            cv2.putText(out, "⚠  DROWSY DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    else:
        # No face detected — draw grey indicator
        cv2.putText(out, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    return out, label, conf


# ── Alarm (Web Audio API) ─────────────────────────────────────────────────────
ALARM_JS = """
<script>
(function() {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    function beep(freq, start, dur) {
        const o = ctx.createOscillator(), g = ctx.createGain();
        o.connect(g); g.connect(ctx.destination);
        o.frequency.value = freq;
        o.type = 'square';
        g.gain.setValueAtTime(0.25, ctx.currentTime + start);
        g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + start + dur);
        o.start(ctx.currentTime + start);
        o.stop(ctx.currentTime + start + dur);
    }
    beep(660, 0.00, 0.2);
    beep(660, 0.25, 0.2);
    beep(880, 0.50, 0.3);
})();
</script>
"""

def play_alarm():
    """Inject JS alarm into the Streamlit page. Fires every 1.5s while drowsy."""
    st.components.v1.html(ALARM_JS, height=0)


# ── UI Layout ─────────────────────────────────────────────────────────────────
st.title("🚗 Driver Drowsiness Detection")
st.markdown("Real-time detection using OpenCV face detection + PyTorch classification.")

with st.sidebar:
    st.header("⚙️ Settings")

    # Model selector — switch between all 3 trained models
    model_choice = st.selectbox("Model", list(MODEL_FILES.keys()))
    model = load_model(model_choice)

    if model is None:
        st.error(f"No saved model for **{model_choice}**.\nTrain it first in the notebook.")
    else:
        st.success(f"{model_choice} loaded ✓")

    st.divider()
    source = st.radio("Input source", ["Webcam", "Upload video"])

    st.divider()
    st.markdown("**Legend**")
    st.markdown("🟡 circles = detected eyes")
    st.markdown("🟩 box = Alert")
    st.markdown("🟥 box = Drowsy")


col_feed, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("📊 Live Stats")
    status_box = st.empty()
    conf_box   = st.empty()
    alert_box  = st.empty()


# ── Inference loop ────────────────────────────────────────────────────────────
def run_inference_loop(frame_source):
    """
    Main frame processing loop.
    Detects face with OpenCV Haar Cascade, runs PyTorch inference,
    draws annotations, and triggers continuous alarm when drowsy.
    """
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

        # Process frame — face detection + inference + annotation
        annotated, label, conf = process_frame(frame, model)

        # Convert BGR → RGB for Streamlit display
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB", use_column_width=True)

        is_drowsy = label == "Drowsy"
        now = time.time()

        # Update stats panel
        status_box.metric("Status", f"{'🔴' if is_drowsy else '🟢'} {label}")
        conf_box.metric("Confidence", f"{conf*100:.1f}%" if conf > 0 else "—")

        if is_drowsy:
            alert_box.error("⚠️ DROWSY! Please pull over!")
            # Replay alarm every 1.5 seconds continuously while drowsy
            if now - last_alarm_time >= 1.5:
                with sound_placeholder:
                    play_alarm()
                last_alarm_time = now
        elif label == "No Face":
            alert_box.warning("👤 No face detected")
        else:
            alert_box.success("✅ Driving normally")


# ── Frame generators ──────────────────────────────────────────────────────────
def webcam_frames():
    """Generator yielding live frames from the default webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam. Check permissions.")
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
    """Generator yielding frames from an uploaded video at ~10fps."""
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
        st.subheader("📷 Webcam Feed")
        if st.button("▶ Start webcam", key="start_webcam"):
            run_inference_loop(webcam_frames())
else:
    with col_feed:
        st.subheader("📁 Upload Dashcam Footage")
        uploaded = st.file_uploader(
            "Upload a video file", type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded and st.button("▶ Analyse video", key="analyse_video"):
            run_inference_loop(video_file_frames(uploaded))