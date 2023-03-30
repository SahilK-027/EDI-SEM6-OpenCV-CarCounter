# =======================================================================================================
# Dependencies
# =======================================================================================================
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# =======================================================================================================
# Creating webcam Object
# =======================================================================================================
cap = cv2.VideoCapture(0)
# Setting Width and height of webcam
cap.set(3, 1280)
cap.set(4, 720)

# video as feed
cap = cv2.VideoCapture('Videos/cars2.mp4')

# =======================================================================================================
# Creating YOLO model
# =======================================================================================================
model = YOLO('yolov8n.pt')
classnames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
             'ring', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
             'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
             'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush' ]
mask = cv2.imread('opencvMask.png')

# =======================================================================================================
# Sort Instance
# =======================================================================================================
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [300, 297, 1080, 297]
totalCnt = []

while True:
    success, img = cap.read()
    imageRegion = cv2.bitwise_and(img, mask)
    results = model(imageRegion, stream=True)

    detections=np.empty((0, 5))

    for i in results:
        boundingBoxes = i.boxes
        for box in boundingBoxes:
            # Drawing bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1

            # Drawing confidence value
            conf = math.ceil(box.conf[0] * 100) / 100

            cls = int(box.cls[0])
            currentClass = classnames[cls]
            if currentClass == 'car' and conf > 0.3:
                # Drawing class name value
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(40, y1)), scale=0.8, thickness=1)

                # Drawing bounding box
                # cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=5)

                # Save to detection list
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                # print(len(detections))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(150, 0, 255))
        # cvzone.putTextRect(img, f'{Id}', (max(0, x1), max(40, y1)), scale=2, thickness=3, offset=10)

        #Counter logic
        centerX, centerY = x1 + w // 2, y1 + h // 2
        if limits[0] < centerX < limits[2] and limits[1] - 20 < centerY < limits[3] + 20:
            if totalCnt.count(Id) == 0:
                totalCnt.append(Id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
        cvzone.putTextRect(img, f'Count : {len(totalCnt)}', (50, 50))

    cv2.imshow('Image', img)
    # cv2.imshow('imageRegion', imageRegion)
    cv2.waitKey(1)