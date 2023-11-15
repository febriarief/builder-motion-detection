import cv2
from ultralytics import YOLO


model = YOLO("best.pt")
file_video = "sample-1.mp4"
cap = cv2.VideoCapture(file_video)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.predict(frame, verbose=False, conf=0.25, imgsz=320)
        annotated_frame = results[0].plot()
        imS = cv2.resize(annotated_frame, (960, 540))
        cv2.imshow("Worker Detection", imS)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
