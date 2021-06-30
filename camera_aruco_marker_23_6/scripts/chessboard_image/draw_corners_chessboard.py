import cv2
import numpy as np

filename = 'distort_1.png'

number_of_squares_X = 10
number_of_squares_Y = 8 
nX = number_of_squares_X - 1
nY = number_of_squares_Y - 1 

def main():
	image = cv2.imread(filename)
	print(image.shape)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	success, corners = cv2.findChessboardCorners(gray, (nX, nY), None)
	if success == True:
		cv2.drawChessboardCorners(image, (nX, nY), corners, success)
		size = len(filename)
		new_filename = filename[:size - 4]
		new_filename = new_filename + '_drawn_corners.jpg'
		cv2.imwrite(new_filename, image)
		cv2.namedWindow('image', cv2.WINDOW_NORMAL)
		cv2.imshow("image", image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

main()
