import sys
from pathlib import Path

# yolov5 directory to the Python path
repo_path = Path('/Users/adityapandey/Desktop/ML/YOLO/yolov5')
sys.path.append(str(repo_path))

import torch
import cv2
import numpy as np

# Load YOLOv5 model from the local repository
model = torch.hub.load(repo_path, 'custom', path=repo_path / 'yolov5s.pt', source='local')

image_path = 'CBlock.jpeg'  # img path
img = cv2.imread(image_path)

results = model(img)

boxes = results.xyxy[0].numpy()  
labels = results.names           

for box in boxes:
    xmin, ymin, xmax, ymax, confidence, class_idx = box
    label = labels[int(class_idx)]
    
    if label == 'car':
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        
        cv2.putText(img, f'{label} {confidence:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

output_path = 'output.jpg'
cv2.imwrite(output_path, img)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()