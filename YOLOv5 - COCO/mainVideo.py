import sys
from pathlib import Path
import torch
import cv2
import numpy as np

repo_path = Path('Yolo/yolov5')
if repo_path not in sys.path:
    sys.path.append(str(repo_path))

model = torch.hub.load(str(repo_path), 'custom', path=repo_path / 'yolov5s.pt', source='local')

video_path = 'Yolo/Dataset.mp4'
output_path = 'Yolo/output_video.mp4'

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    boxes = results.xyxy[0].numpy()
    labels = results.names

    for box in boxes:
        xmin, ymin, xmax, ymax, confidence, class_idx = box
        label = labels[int(class_idx)]

        if label == 'car':
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()