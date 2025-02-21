from ultralytics import YOLO
import cv2
import numpy as np
from cvzone import cornerRect, putTextRect
from math import ceil
from sort import * 

cap = cv2.VideoCapture("images/car1.mp4")

model = YOLO("yolo weights/yolov8n.pt")
limits = [423, 507, 1123, 507]
total_count = []

classNames = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
            19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
            28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
            53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
            62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
            71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'}

mask = cv2.imread("images\car1 mask.png")
tracker = Sort(max_age = 20)

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream = True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            confidence = ceil((box.conf[0] * 100)) / 100
            name = classNames[int(box.cls[0])]
            if (name == "car" or name == "truck" or name == "bicycle" or name == "motorcycle" or name == "bus") and confidence > 0.3:
                #putTextRect(img, f'{name} {confidence}', (max(0, x1), max(35, y1)))
                #cornerRect(img, (x1, y1, w, h))
                current_array = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, current_array))

    result_tracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 3)

    for result in result_tracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2, Id = int(x1), int(y1), int(x2), int(y2), int(Id)
        w, h = x2 - x1, y2 - y1
        cornerRect(img, (x1, y1, w, h), colorR = (255, 0, 0))
        putTextRect(img, f'{Id}', (max(0, x1), max(35, y1)))

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if(limits[0] < cx < limits[2]) and (limits[1] - 20 < cy < limits[3] + 20):
            if total_count.count(Id) == 0 :
                total_count.append(Id)
    putTextRect(img, f'Car Count : {len(total_count)}', (50, 50))  
    cv2.imshow("Image", img)
    #cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)