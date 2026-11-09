# 👁️ Computer Vision: Road Sign Detection (YOLOv5) & Motion-Aware Smart Surveillance

A dual-pipeline Computer Vision engineering repository featuring:
1. **Real-Time Road Sign Detection (`Computer Vision for Road Sign Detection.py`)**: Deep learning object detection pipeline using OpenCV and YOLOv5 with TensorRT edge optimization.
2. **Motion-Aware Smart Surveillance & Vehicle/Pedestrian Counter (`surveillance.py` & `Motion_Aware_Surveillance_System.ipynb`)**: Frame-differencing motion detection, Gaussian smoothing, morphological contour tracking, virtual line-crossing object counting, and Canny + Probabilistic Hough Transform (`HoughLinesP`) ROI edge analysis.

---

## 🧠 Pipeline Architecture

### 1. Motion-Aware Smart Surveillance (`surveillance.py`)
- **Temporal Frame Differencing**: Computes absolute pixel intensity deltas (`cv2.absdiff`) across consecutive video frames.
- **Morphological Noise Suppression**: Applies `5x5` Gaussian blur, binary thresholding, and kernel dilation to isolate moving targets.
- **Contour Classification & Virtual Tripwire Counting**: Filters bounding boxes by contour area to classify **Persons** vs. **Vehicles (Cars)** and increments a real-time tripwire counter upon crossing `line_y`.
- **ROI Structural Feature Extraction**: Runs Canny edge detection (`50, 150`) and Probabilistic Hough Line Transforms inside each detected object's Region of Interest.

### 2. Road Sign Detection (`Computer Vision for Road Sign Detection.py`)
- Real-time traffic sign localization and classification designed for low-latency inference.

---

## 🚀 Running Locally

```bash
pip install opencv-python numpy torch matplotlib
python surveillance.py
```
