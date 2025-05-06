

import torch
from pathlib import Path
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5_custom.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated_frame = results.render()[0]
    cv2.imshow('YOLOv5 Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()