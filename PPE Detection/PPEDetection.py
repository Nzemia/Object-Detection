from ultralytics import YOLO
import cv2
import cvzone
import math


# for webcam
#cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)



cap = cv2.VideoCapture("../Videos/ppe-1-1.mp4")
# cap = cv2.VideoCapture(0)




model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

myColor = (0, 0, 255)

while True:
    success, image = cap.read()
    results = model(image, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(image, (x1 ,y1, w, h))

            # confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)


            # Class
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.5:
                if currentClass == "NO-Hardhat" or currentClass == "NO-Mask" or currentClass == "NO-Safety Vest":
                    myColor=(0, 0, 255)
                elif currentClass == "Hardhat" or currentClass == "Mask" or currentClass == "Safety Vest":
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)


                cvzone.putTextRect(image, f"{classNames[cls]}" f"{conf}",
                                   (max(0, x1), max(35, y1)), scale=0.7,
                                   thickness=1, colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)

                cv2.rectangle(image, (x1, y1), (x2, y2), myColor, 3)



    cv2.imshow("Webcam", image)
    cv2.waitKey(1)