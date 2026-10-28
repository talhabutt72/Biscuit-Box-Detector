# Biscuit Box Detector

An object detection project that finds and classifies biscuit boxes in photos using YOLO11n. Built end to end: data collection, labeling, training on Google Colab, and a Streamlit app for testing.

> **Status: baseline.** The dataset is small (49 photos from one session), so this is a working pipeline, not a reliable model. See [Limitations](#limitations).

## Classes

The model detects 5 biscuit box types:

`candi`, `lemon`, `nan-khatai`, `rio`, `zeera-plus`

A sixth class, `sooper`, was labeled originally but dropped. Its red packaging looked too similar to `candi`, and the model kept confusing the two. With this little data, dropping it was the honest fix, so the model will not recognize sooper boxes correctly.

## Project workflow

1. **Data collection:** photos of biscuit boxes taken at a friend's shop, 4-7 shots per arrangement from different angles.
2. **Labeling:** bounding boxes drawn in Label Studio, exported in YOLO format.
3. **Splitting:** split by time, not randomly (see below).
4. **Training:** YOLO11n fine-tuned from pretrained weights on a Colab T4 GPU.
5. **Demo:** Streamlit app with image upload and camera snapshot.

## Dataset

- 49 images, 5 classes after dropping `sooper`
- Single session, same background and lighting
- Images are full-resolution phone photos, resized to 640 px by YOLO during training
- Split: roughly 40 train / 9 val (`[TODO: fill in the exact counts from your run]`)
- No separate test set yet (too few images to hold one out meaningfully)

### Why the split is by time

Several photos were taken of each arrangement, so they are near-duplicates. A random split would put some of them in train and some in val, and the val score would be inflated because the model had effectively seen those scenes.

Instead, the photos were sorted by the timestamp in their filenames (`IMG_YYYYMMDD_HHMMSS_...`), and a single cut was made at the longest pause between photos near the 40th image. Everything before the cut is train, everything after is val.

This is imperfect: the pauses between arrangements were not always clear, so some leakage near the cut is possible. It is fine for a baseline, not for a final evaluation.

## Training

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # pretrained weights
model.train(
    data="/content/dataset/data.yaml",
    epochs=150,
    imgsz=640,
    batch=8,
    patience=30,   # stop if val does not improve for 30 epochs
    project="/content/drive/MyDrive/yolo_runs",
    name="run2",
)
```

Dataset layout expected by YOLO:

```
dataset/
  images/train  images/val
  labels/train  labels/val
  data.yaml
```

## Results

First run (6 classes, before dropping `sooper`), on a 13-image val set:

| Metric | Value |
|---|---|
| mAP50 | 0.693 |
| mAP50-95 | 0.419 |
| Precision | 0.506 |
| Recall | 0.745 |

Per-class numbers for `lemon` and `rio` (0.995 mAP50) came from only 2 val boxes each and mean very little. `sooper` was the weakest class (0.246 mAP50) and was mostly confused with `candi`.

Second run (5 classes): `[TODO: paste the final mAP50 and mAP50-95 from your run2 results table]`

Treat all of these as rough indicators. The val set is tiny, comes from the same session as the training data, and `best.pt` was selected using it.

## Run the Streamlit app

1. Put `best.pt` (the weights from `run2/weights/`) next to `app.py`.
2. Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app has two input modes: upload an image, or take a snapshot with the camera. It shows the annotated image, the detected classes with confidence scores, and a confidence-threshold slider.

`requirements.txt`:

```
streamlit
ultralytics
opencv-python-headless
```

Use `opencv-python-headless` rather than `opencv-python`, which fails on Streamlit Cloud.

## Deploying on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt` and `best.pt` to a GitHub repo. The weights file is about 5.5 MB, so Git LFS is not needed.
2. Go to share.streamlit.io and sign in with GitHub.
3. Create a new app, select the repo, and set the main file to `app.py`.
4. Deploy. The first build takes a few minutes while it installs ultralytics and torch.

## Limitations

- **Small dataset:** 49 images, about 10 distinct arrangements. Metrics are noisy and should not be quoted as real performance.
- **One session:** same lighting and background throughout, so the model may fail on new settings.
- **Optimistic validation:** val comes from the same session and was used to pick the best checkpoint.
- **Class confusion:** boxes with similar packaging (like `candi` and the dropped `sooper`) get mixed up.
- **Snapshot only:** the app predicts on one photo at a time, not live video.

## What I would do next

- Take 150-200 more photos with a new arrangement almost every time, plus varied angles, lighting and backgrounds
- Hold out a proper test set of fresh photos that never touch training or validation
- Re-add `sooper` once there are enough photos of it next to `candi`
- Check the confusion matrix and the wrong predictions from the app to decide which photos to collect
- Try live video detection with streamlit-webrtc or an OpenCV script

## Tools

Python, Ultralytics YOLO11, Label Studio, Google Colab, Streamlit.

## Author

`[TODO: your name and links]`
