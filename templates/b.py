import cv2
import numpy as np

# Load the image
from ultralytics import YOLO
model=YOLO("C:/Users/uditpc/Pictures/deep_learning/project/train/weights/best.pt")
# Define the list of triangles with their coordinates
# triangles = [[[76, 164.125], [190, 114.125], [61, 88.125]], [[43, 137.125], [174, 21.125], [140, 133.125]], [[63, 567.125], [63, 529.125], [135, 537.125]], [[645, 471.125], [620, 511.125], [746, 511.125]], [[709, 74.125], [638, 72.125], [701, 107.125]]]
cap = cv2.VideoCapture(0)  
while True:
    success,frame1 = cap.read()
    if not success:
        break
    if success:
        result=model.predict(frame1,conf=0.60)
        frame_cp=result[0].plot()
        # for triangle_coords in triangles:
        #     print("this "+str(triangle_coords))
        #     pts = np.array(triangle_coords, dtype=np.int32)
        #     cv2.fillPoly(frame_cp, [pts], (255, 255, 255))
            # Run YOLOv8 inference on the frame
        # results = model.predict(frame1,conf=0.60)
        # for res in results[0]:
        #     if res.boxes.cls[0].item()==1.0:
        #         jacket=True;
        #     if res.boxes.cls[0].item()==0.0:
        #         halmet=True;
        # annotated_frame = results[0].plot()

        cv2.imshow("this",frame_cp)
            # Break the loop if'q' is pressed
        key=cv2.waitKey(100) 
        if key == ord("q"):
            break
    else:
        break

