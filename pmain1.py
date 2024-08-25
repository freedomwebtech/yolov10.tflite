import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone


model = YOLO("")  
 
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)


cap=cv2.VideoCapture('t.mp4')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count=0
while True:
    ret,frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
       break
    frame = cv2.resize(frame, (1020, 600))

    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
   
    cv2.imshow("RGB", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


