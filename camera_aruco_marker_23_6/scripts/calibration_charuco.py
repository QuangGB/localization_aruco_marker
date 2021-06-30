import numpy
import cv2
from cv2 import aruco
import argparse
import pickle
import glob
import os

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to the calibration video")
ap.add_argument("-c", "--captures", required=True,
	help="minimum number of valid captures required", type=int)
args = vars(ap.parse_args())

CHARUCOBOARD_ROWCOUNT = 8
CHARUCOBOARD_COLCOUNT = 6 
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = aruco.CharucoBoard_create(
	squaresX=CHARUCOBOARD_COLCOUNT,
	squaresY=CHARUCOBOARD_ROWCOUNT,
	squareLength=0.04,
	markerLength=0.02,
	dictionary=ARUCO_DICT)

# Corners discovered in all images processed
corners_all = []

# Aruco ids corresponding to corners discovered 
ids_all = [] 

# Determined at runtime
image_size = None 

# This requires a video taken with the camera you want to calibrate
cap = cv2.VideoCapture(args["video"])

# The more valid captures, the better the calibration
validCaptures = 0

while cap.isOpened():
	ret, img = cap.read()
	if ret is False:
		break

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	corners, ids, _ = aruco.detectMarkers(image=gray, dictionary=ARUCO_DICT)

	if ids is None:
		continue

	img = aruco.drawDetectedMarkers(image=img, corners=corners)

	response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
		markerCorners=corners,
		markerIds=ids,
		image=img,
		board=CHARUCO_BOARD)
	if response > 20:
		corners_all.append(charuco_corners)
		ids_all.append(charuco_ids)

		img = aruco.drawDetectedCornersCharuco(
			image=img,
			charucoCorners=charuco_corners,
			charucoIds=charuco_ids)
		if not image_size:
			image_size = gray.shape[::-1]

		proportion = max(img.shape)/1000.0
		img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))

		cv2.imshow('charuco board', img)
		if cv2.waitKey(0) == ord('q'):
			break 

		validCaptures += 1
		if validCaptures == args["captures"]:
			break 

cv2.destroyAllWindows()

# Show number of valid captures
print("{} valid captures".format(validCaptures))

if validCaptures < args["captures"]:
	print("Calibration was unsuccessful. We couldn't detect enough charucoboards in the video.")
	print("Perform a better capture or reduce the minimum number of valid captures required.")
	exit()

# Make sure we were able to calibrate on at least one charucoboard
if len(corners_all) == 0:
	print("Calibration was unsuccessful. We couldn't detect charucoboards in the video.")
	print("Make sure that the calibration pattern is the same as the one we are looking for (ARUCO_DICT).")
	exit()
print("Generating calibration...")

calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
	charucoCorners=corners_all,
	charucoIds=ids_all,
	board=CHARUCO_BOARD,
	imageSize=image_size,
	cameraMatrix=None,
	distCoeffs=None)

print('camera parameters matrix:\n{}'.format(cameraMatrix))
print('\ncamera distortion coefficient:\n{}'.format(distCoeffs))

f = open('./cameraCalibration1.pckl', 'wb')
pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)
f.close()

print('Calibrate successful. Calibration file created:\n{}'.format('cameraCalibration1.pckl'))


