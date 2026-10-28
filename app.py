import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Biscuit Box Detector", layout="centered")
st.title("Biscuit Box Detector")

@st.cache_resource
def load_model():
    return YOLO("best.pt")
model = load_model()

conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

mode = st.radio("Input", ["Upload image", "Use camera"], horizontal=True)

if mode == "Upload image":
file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
else:
file = st.camera_input("Take a photo")

if file is not None:
image = Image.open(file).convert("RGB")
results = model.predict(np.array(image), conf=conf)
annotated = results[0].plot()[:, :, ::-1] # BGR -> RGB
st.image(annotated, caption="Detections", use_container_width=True)

boxes = results[0].boxes
if len(boxes) == 0:
st.info("Nothing detected. Try lowering the confidence slider.")
else:
for b in boxes:
st.write(f"{model.names[int(b.cls)]}: {float(b.conf):.2f}")