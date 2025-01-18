from ultralytics import YOLO
import cv2
import cvzone
import math

from sort import *

cap = cv2.VideoCapture("../Videos/cars.mp4")

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


mask = cv2.imread("masked_design.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# create a line for counting
limits = [400, 297, 673, 297]
totalCount = []

while True:
    success, image = cap.read()
    #results = model(image, stream=True)


    # overlay the mask on the image
    imageRegion = cv2.bitwise_and(image, mask)

    imageGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    image = cvzone.overlayPNG(image, imageGraphics, [0, 0])

    results = model(imageRegion, stream=True)

    detections = np.empty((0, 5))


    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1

            # confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)


            # Class
            cls = int(box.cls[0])

            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus"\
                    or currentClass== "motorbike" and conf > 0.3:
                # cvzone.putTextRect(image, f"{currentClass}" f"{conf}", (max(0, x1), max(35, y1)),
                #                scale=0.7, thickness=1, offset=3)
                # cvzone.cornerRect(image, (x1, y1, w, h), l=10, rt=5)

                # save append the detections
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))


    resultsTracker = tracker.update(detections)

    # drawing the line
    # cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # loop through the results
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(image, (x1, y1, x2-x1, y2-y1), l=9, rt=2, colorR=(255, 0, 0))
        # cvzone.putTextRect(image, f"{int(id)}", (max(0, x1), max(35, y1)),
        #                    scale=2, thickness=3, offset=10)

        # get the center of the bounding box
        cx,cy = x1 + w//2, y1 + h//2
        # cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED )

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # cvzone.putTextRect(image, f"Count: {len(totalCount)}", (50, 50))

        cv2.putText(image, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 50, 255), 8)



    cv2.imshow("Image", image)
    # cv2.imshow("Image Region", imageRegion)
    cv2.waitKey(1)