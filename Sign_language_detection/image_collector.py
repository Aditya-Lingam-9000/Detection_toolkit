import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

data_dir = 'CV_Projects\Sign_language_detection\data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


no_of_classes = 3

dataset_size = 100

cap = cv2.VideoCapture(3)

for j in range(no_of_classes):
    if not os.path.exists(os.path.join(data_dir,str(j))):
        os.makedirs(os.path.join(data_dir,str(j)))
    
    print(f"Collecting the data for class {j}")

    done = False
    while True:
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        cv2.putText(frame, 'Ready? Press "S"(start)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('s'):
            break

    counter = 0
    while counter < dataset_size:
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(data_dir,str(j),'{}.jpg'.format(counter)),frame)
        counter+=1

cap.release()
cv2.destroyAllWindows()