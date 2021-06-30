#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import pickle
import cv2.aruco as aruco 

#check data file after calibration
if not os.path.exists('D:\localization_aruco\chessboard_image\cameraCalibration_chessboard.pckl'):
  print('You need to calibrate your camera')
  exit()
else:
  f = open('D:\localization_aruco\chessboard_image\cameraCalibration_chessboard.pckl', 'rb')
  (cameraMatrix, distCoeffs, _, _) = pickle.load(f)
  f.close()
  if cameraMatrix is None or distCoeffs is None:
    print('You need remove file pckl and recalibrate')
    exit()
img = cv2.imread('distort_20.png')

w, h = 1920, 1080
newcamera_matrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, newcamera_matrix)

def aruco_marker_detector(img):
  ARUCO_PARAMETER = aruco.DetectorParameters_create()
  ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_img, ARUCO_DICT, parameters=ARUCO_PARAMETER)
  #draw outline if marker is found 
  if ids is not None: 
    ids = ids.flatten()
    for _id, corner in zip(ids, corners):
      corner = corner.reshape((4,2))
      (topLeft, topRight, bottomRight, bottomLeft) = corner

      #convert to (x, y) coordinate
      topLeft = (int(topLeft[0]), int(topLeft[1]))
      topRight = (int(topRight[0]), int(topRight[1]))
      bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
      bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

      #draw line bouding box of aruco detected
      cv2.line(img, topLeft, topRight, (0, 0, 255), 2)
      cv2.line(img, topRight, bottomRight, (0, 0, 255), 2)
      cv2.line(img, bottomRight, bottomLeft, (0, 0, 255), 2)
      cv2.line(img, bottomLeft, topLeft, (0, 0, 255), 2) 

      #calculate and draw center point of (x, y) coordinate
      cX = int((topLeft[0] + bottomRight[0]) / 2)
      cY = int((topLeft[1] + bottomRight[1]) / 2)
      cv2.circle(img, (cX, cY), 4, (0, 255, 0), -1)
      #print('center of marker: \n{}'.format((cX, cY)))
      #print('corner: \n{}'.format(corner))

      #calculate width of bounding box
      dX = topRight[0] - topLeft[0]
      dY = topRight[1] - topLeft[1]
      print('chieu dai canh tinh theo x:\n{}'.format(dX))
      print('chieu dai canh tinh theo y:\n{}'.format(dY))

      #draw ID on image
      font = cv2.FONT_HERSHEY_SIMPLEX
      str1 = str('id is: ') + str(_id)
      cv2.putText(img, str1, (topLeft[0], topLeft[1] - 15), font, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
      cv2.putText(img, str('center of marker'), (cX -15, cY - 15), font, 0.9, (255, 0, 255), 2, cv2.LINE_AA)  
  return ids, corners
ids, corners = aruco_marker_detector(dst)

'''
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcamera_matrix, (img.shape[:2]), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)'''
cv2.namedWindow('undistort image', cv2.WINDOW_NORMAL)
cv2.imshow("undistort image", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

