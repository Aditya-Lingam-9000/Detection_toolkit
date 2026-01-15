import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsv_color = cv2.cvtColor(c,cv2.COLOR_BGR2HSV)
    hue = hsv_color[0][0][0]
    if hue>=165: # Red color case if hue is near to 180 then we have to split the range
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
    elif hue <= 15:  # Red color case if hue is near to 0 then we have to split the range
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else: # General case, what it does is to take 10 values below and above the hue value
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    
    return lowerLimit,upperLimit



col = [0,0,255] # BGR format
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lowerlimit,upperlimit = get_limits(color=col)
    mask = cv2.inRange(hsv_frame,lowerlimit,upperlimit) # creates a mask with the specified limits
    # if the specified color is detected then the pixel value is set to 255 else to 0
    # if we want to check how the mask looks like we can use cv2.imshow('mask',mask)
    

    # now we will use bitwise and to extract the color part from the frame
    # result = cv2.bitwise_and(hsv_frame,hsv_frame,mask=mask) # it will keep the pixel value where mask is 255 else set to 0
    # just for visualization purpose we will convert the result back to BGR
    # result = cv2.cvtColor(result,cv2.COLOR_HSV2BGR)

    mask_array = Image.fromarray(mask) # converts the mask into a PIL image because findcontours function in OpenCV works on binary images
    bbox = mask_array.getbbox() # returns the bounding box of the non zero regions in the image where the color is detected
    if bbox:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    if not ret:
        break
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break