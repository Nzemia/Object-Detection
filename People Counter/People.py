from ultralytics import YOLO
import cv2
import cvzone
import math

from sort import *

cap = cv2.VideoCapture("../Videos/people.mp4")


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

mask = cv2.imread("people_mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# create a line for counting
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

totalCountUp = []
totalCountDown = []

while True:
    success, image = cap.read()
    #results = model(image, stream=True)


    # overlay the mask on the image
    imageRegion = cv2.bitwise_and(image, mask)

    imageGraphics = cv2.imread("graphics-1.png", cv2.IMREAD_UNCHANGED)
    image = cvzone.overlayPNG(image, imageGraphics, [730, 260])

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

            if currentClass == "person" and conf > 0.3:
                # cvzone.putTextRect(image, f"{currentClass}" f"{conf}", (max(0, x1), max(35, y1)),
                #                scale=0.7, thickness=1, offset=3)
                # cvzone.cornerRect(image, (x1, y1, w, h), l=10, rt=5)

                # save append the detections
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))


    resultsTracker = tracker.update(detections)

    # drawing the lines
    cv2.line(image, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(image, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

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

        # up counter
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 20 < cy < limitsUp[1] + 20:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(image, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        # down counter
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 20 < cy < limitsDown[1] + 20:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(image, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)


        # cvzone.putTextRect(image, f"Count: {len(totalCount)}", (50, 50))

        # counter for the people going up
        cv2.putText(image, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_SIMPLEX, 3, (139, 195, 75), 6)

        # counter for the people going down
        cv2.putText(image, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_SIMPLEX, 3, (50, 50, 230), 6)



    cv2.imshow("Image", image)
    # cv2.imshow("Image Region", imageRegion)
    cv2.waitKey(1)