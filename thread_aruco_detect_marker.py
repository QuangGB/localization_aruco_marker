#!/usr/bin/env python

import rospy
from imutils.video import FPS
from imutils.video import FileVideoStream
import imutils
import time
import argparse
import numpy as np
import cv2
import cv2.aruco as aruco
from math import sqrt, acos, pi
import sys
import os 
import pickle
from math import *
from camera_aruco_marker_23_6.msg import data  


resize_factor = 2

pixel_per_mm = 4.5

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to input video file")
args = vars(ap.parse_args())

print("[INFO] starting video file thread...")
fvs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

fps = FPS().start()

#check data file after calibration
if not os.path.exists('/home/quangle/aruco_localization_ws/src/camera_aruco_marker_23_6/scripts/chessboard_image/cameraCalibration_chessboard.pckl'):
	#/home/quangle/aruco_localization_ws/src/camera_aruco_marker_23_6/scripts/chessboard_image
	print('You need to calibrate your camera')
	exit()
else:
	f = open('/home/quangle/aruco_localization_ws/src/camera_aruco_marker_23_6/scripts/chessboard_image/cameraCalibration_chessboard.pckl', 'rb')
	(cameraMatrix, distCoeffs, _, _) = pickle.load(f)
	f.close()
	if cameraMatrix is None or distCoeffs is None:
		print('You need remove file pckl and recalibrate')
		exit() 
#print('cameraMatrix:\n{}'.format(cameraMatrix))
#print('distCoeffs:\n{}'.format(distCoeffs))

def aruco_marker_detector(img):
	ARUCO_PARAMETER = aruco.DetectorParameters_create()
	ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_img, ARUCO_DICT, parameters=ARUCO_PARAMETER)
	id_list = []
	#draw outline if marker is found 
	if ids is not None: 
		ids = ids.flatten()
		for _id, corner in zip(ids, corners):
			try:
				if ids.size != 0:
					id_list.append(_id)
			except:
				print('ID list empty')

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

			#draw ID on image
			font = cv2.FONT_HERSHEY_SIMPLEX
			str1 = str('id is: ') + str(_id)
			cv2.putText(img, str1, (topLeft[0], topLeft[1] - 15), font, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
			cv2.putText(img, str('center of marker'), (cX -15, cY - 15), font, 0.9, (255, 0, 255), 2, cv2.LINE_AA)	
	return ids, corners, id_list

def center_of_image(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#gray = cv2.resize(gray, (300, 400))
	(h, w) = gray.shape[:2]
	(cX_img, cY_img) = (w // 2, h // 2)
	cv2.circle(img, (cX_img, cY_img), 8, (255, 255, 0), -1)
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img, str('center of image'), (cX_img - 15, cY_img - 15), font, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
	#print('center of image: \n{}'.format((cX_img, cY_img)))
	return cX_img, cY_img

def distance(img, cX_img, cY_img, corners):
	distances = []
	#cX_img, cY_img = center_of_image(img)
	#_, corners = aruco_marker_detector(img)
	for corner in corners:
		corner = corner.reshape((4, 2))
		cX = int((corner[0][0] + corner[2][0]) / 2)
		cY = int((corner[0][1] + corner[2][1]) / 2)
		cv2.line(img, (cX, cY), (cX_img, cY_img), (255, 255, 255), 1)
		distance = sqrt((cX - cX_img)**2 + (cY - cY_img)**2)
		str1 = str(distance)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img, str1, (int((cX + cX_img) / 2), int((cY + cY_img) / 2)), font, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
		distances.append(distance)
		#print('distance is: \n{}'.format(distance))
	return distances

def calculate_position_orientation(img, distances, cX_img, cY_img, corners):
	x_poss = []
	y_poss = []
	thetas = []
	#distances = distances.reshape((-1, 1))
	for distance, corner in zip(distances, corners):
		corner = corner.reshape((4, 2))
		cX = int((corner[0][0] + corner[2][0]) / 2)
		cY = int((corner[0][1] + corner[2][1]) / 2)
		x_pos = cX - cX_img
		y_pos = cY - cY_img
		#covert pixel to mm
		x_pos = round((x_pos / pixel_per_mm), 2)
		y_pos = round((y_pos / pixel_per_mm), 2) 
		theta = acos(x_pos / distance) * 180 / pi
		x_poss.append(x_pos)
		y_poss.append(y_pos)
		thetas.append(theta)
	return x_poss, y_poss, thetas

def main():
	#cap = cv2.VideoCapture('rtsp://admin:BTNJZR@192.168.1.190:554/H.264')
	#cap.set(3, 1920)
	#cap.set(4, 1080)
	rospy.init_node('aruco_marker_23_6', anonymous=True)
	data_pub = rospy.Publisher('data_from_aruco_marker', data, queue_size=10)
	rate = rospy.Rate(10)
	msg = data()

	while fvs.more():
		distances = []
		x_poss = []
		y_poss = []
		thetas = []
		img = fvs.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#print(img.shape)
		h, w = gray.shape
		#optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
		#undistorted_img = cv2.undistort(img, cameraMatrix, distCoeffs, None, optimal_camera_matrix)
		ids, corners, id_list = aruco_marker_detector(img)
		cX_img, cY_img = center_of_image(img)
		distances = distance(img, cX_img, cY_img, corners)
		x_poss, y_poss, thetas = calculate_position_orientation(img, distances, cX_img, cY_img, corners)

		#print('distance = \n{}'.format(distances))
		#print('ids: = \n{}'.format(ids))
		#print('x_pos = \n{}'.format(x_poss))
		#print('y_pos = \n{}'.format(y_poss))
		#print('theta = \n{}'.format(thetas))

		id_arr = np.array(id_list)
		x_poss_arr = np.array(x_poss)
		y_poss_arr = np.array(y_poss)
		thetas_arr = np.array(thetas)
		
		try:
			if id_arr.size != 0:
				msg.ids = id_arr[0]
		except:
			print("can not find id")
		try:
			if id_arr.size != 0:
				msg.x_poss = x_poss_arr[0]
		except:
			print("can not find id")
		try:
			if id_arr.size != 0:
				msg.y_poss = y_poss_arr[0]
		except:
			print("can not find id")
		try:
			if id_arr.size != 0:
				msg.thetas = thetas_arr[0]
		except:
			print("can not find id")

		data_pub.publish(msg)

		cv2.putText(img, "Queue Size: {}".format(fvs.Q.qsize()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
		cv2.namedWindow('current image', cv2.WINDOW_NORMAL)
		h = int(h / resize_factor)
		w = int(w / resize_factor)
		cv2.resizeWindow('current_image', w, h)
		'''cv2.imshow('current image', img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break'''
		fps.update()

		#stop the timer and display FPS information
		fps.stop()
		#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		fps_val = fps.fps()
		#str_fps = str('FPS: ') + str(fps_val)
		str_fps = "FPS: {:.2f}".format(fps.fps())
		cv2.putText(img, str_fps, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.imshow('current image', img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()
	fvs.stop()

if __name__=='__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass 


