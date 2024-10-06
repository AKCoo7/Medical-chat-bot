import cv2 as cv
import numpy as np


cam = cv.VideoCapture(0)

lower_red = np.array([0,125,125])
upper_red = np.array([10,255,255])

while True:
    ret, frame = cam.read()
    frame = cv.flip(frame,1)

    w = frame.shape[1]
    h = frame.shape[0]
    

    #Smoothen image
    image_smooth = cv.GaussianBlur(frame,(7,7),0)

    # define roi
    mask = np.zeros_like(frame)

    mask[50:350, 50:350] = [255,255,255]

    image_roi = cv.bitwise_and(image_smooth, mask)
    cv.rectangle(frame,(50,50), (350,350), (0,0,255), 2)
    cv.line (frame, (150,50), (150,350), (0,0,255),1)
    cv.line (frame,(250,50), (250,350), (0,0,255),1)
    cv.line (frame,(50,150), (350,150), (0,0,255),1)
    cv.line (frame, (50,250), (350,250), (0,0,255),1)


    #Thresholding
    image_hsv = cv.cvtColor(image_smooth,cv.COLOR_BGR2HSV)
    image_threshold = cv.inRange(image_hsv,lower_red, upper_red)

    # Find Contours
    contours, heirarchy = cv.findContours(image_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Find the index of the largest contour
    if (len(contours) != 0):
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x_bound, y_bound, w_bound, h_bound = cv.boundingRect(cnt)
        cv.rectangle(frame, (x_bound, y_bound), (x_bound + w_bound, y_bound + h_bound), (255,0,0),2)


    # Pointer on video
    M = cv.moments(cnt)
    if(M['m00'] !=0):
        cx = int (M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv.circle (frame, (cx, cy), 4, (0,255,0), -1)

     #Cursor Motion
    if cx in range (150,250):
         if cy < 150:
            print ("Upper Middle")
         elif cy >250:
              print ("Lower Middle")
         else:
            print ("Center")
    if cy in range (150,250):
      if cx < 150:
         print ("Left Middle")
      elif cx > 250:
          print('Right Middle')
      else :
          print('Center')

    cv.imshow('Camera', frame)
    
    key = cv.waitKey(100)
    if key == 27:
        break

cam.release()
cv.destroyAllWindows()
