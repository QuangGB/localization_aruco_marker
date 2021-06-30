'''# Brief description:
# ArUco pose estimation gives us the relative position of the marker and the
# camera: Rotation vector 'rvecs' and translation vector 'tvecs'.
# They represent the transform from the marker to the camera.
# 
# I want to convert the pose data to get the position of the camera in panda
# world (for now we assume the marker is at position 'x=0, y=0, z=0' in panda
# world, to make things simpler), so I can apply the pose to Panda camera
# (to mimic the pose of the real world user camera)

# Pose estimation (ArUco board):
retval, self.rvecs, self.tvecs = aruco.estimatePoseBoard( corners, ids, self.board0, self.cameraMatrix, self.distCoeffs, None, None )

# Convert rotation vector to a rotation matrix:
mat, jacobian = cv.Rodrigues(self.rvecs) # cv.Rodrigues docs: https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
# Transpose the matrix (following approach found at stackoverflow):
mat = cv.transpose(mat) # cv.transpose docs: https://docs.opencv.org/master/d2/de8/group__core__array.html#ga46630ed6c0ea6254a35f447289bd7404
# Invert the matrix (following approach found at stackoverflow, supposed to convert pose data from marker coordinate space to camera coordinate space): 
retval, mat = cv.invert(mat) # cv.invert docs: https://docs.opencv.org/master/d2/de8/group__core__array.html#gad278044679d4ecf20f7622cc151aaaa2

# Create panda matrix so we can apply the data to a node via '.setMat()':
mat3 = Mat3() 
mat3.set(mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1], mat[1][2], mat[2][0], mat[2][1], mat[2][2] )
mat4 = Mat4(mat3)
 
# From here on, pretty much all the values are assigned by trial-and-error, to see what works and what doesn't.

# ROTATION:
# Apply pose estimation rotation matrix to a dummy node:
panda.matrixNode.setMat( mat4 ) 
# Apply the matrixNode HPR to another dummy node with some trial-and-error modifications:
# H=zRot, P=xRot, R=yRot
panda.transNode.setH( - panda.matrixNode.getR() ) # zRot
panda.transNode.setP( panda.matrixNode.getP() + 180 ) # xRot
panda.transNode.setR( - panda.matrixNode.getH() ) # yRot

# TRANSLATION:
# Place dummy node to a marker position in panda world coordinates (marker is at 0,0,0)
panda.transNode.setPos(0, 0, 0) 
# Use translation data from pose estimation:
xTrans = self.tvecs[0][0]
yTrans = self.tvecs[1][0]
zTrans = self.tvecs[2][0]
# Apply translation with negative values, seems to work (trial-and-error):
panda.transNode.setPos(panda.transNode, -xTrans, -zTrans, yTrans)

# Assign values to be applied to panda camera position:
camX = panda.transNode.getX()
camY = panda.transNode.getY()
camZ = panda.transNode.getZ()
camH = panda.transNode.getH()
camP = panda.transNode.getP()
camR = panda.transNode.getR()'''


import numpy as np
import cv2 as cv
'''
rvec = np.array([[1.0],[2.0],[3.0]])
print(rvec.shape)
r_matrix, jacobian = cv.Rodrigues(rvec)
print('r_matrix is:\n{}'.format(r_matrix))
t_matrix = np.array([[1],[2],[3]])
print('t_matrix:\n{}'.format(t_matrix))

A= np.zeros((4, 4))
print(A)
A = np.concatenate((r_matrix, t_matrix), axis=1)
print(A)
print(A[1])
'''

# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: prpject
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-02-04 16:03:01
# @url    : https://www.jianshu.com/p/c5627ad019df
# --------------------------------------------------------
"""
'''import sys
import os
from tools import image_processing
 
sys.path.append(os.getcwd())
import numpy as np
from modules.utils_3d import vis
 
camera_intrinsic = {
    # R, rotation matrix
    "R": [[-0.91536173, 0.40180837, 0.02574754],
          [0.05154812, 0.18037357, -0.98224649],
          [-0.39931903, -0.89778361, -0.18581953]],
         # t, translation vector
    "T": [1841.10702775, 4955.28462345, 1563.4453959],
         # Focal length, f/dx, f/dy
    "f": [1145.04940459, 1143.78109572],
         # principal point, the principal point, the intersection of the principal axis and the image plane
    "c": [512.54150496, 515.45148698]
 
}
 
 
class Human36M(object):
    @staticmethod
    def convert_wc_to_cc(joint_world):
        """
                 World Coordinate System -> Camera Coordinate System: R * (pt-T)
        :return:
        """
        joint_world = np.asarray(joint_world)
        R = np.asarray(camera_intrinsic["R"])
        T = np.asarray(camera_intrinsic["T"])
        joint_num = len(joint_world)
                 # World coordinate system -> camera coordinate system
        # [R|t] world coords -> camera coords
        joint_cam = np.zeros((joint_num, 3))  # joint camera
        for i in range(joint_num):  # joint i
            joint_cam[i] = np.dot(R, joint_world[i] - T)  # R * (pt - T)
        return joint_cam
 
    @staticmethod
    def __cam2pixel(cam_coord, f, c):
        """
                 Camera coordinate system -> pixel coordinate system: (f / dx) * (X / Z) = f * (X / Z) / dx
        cx,ppx=260.166; cy,ppy=205.197; fx=367.535; fy=367.535
                 The formula for mapping from 3D (X, Y, Z) to 2D pixel coordinates P (u, v) is:
        u = X * fx / Z + cx
        v = Y * fy / Z + cy
        D(v,u) = Z / Alpha
        =====================================================
        camera_matrix = [[428.30114, 0.,   316.41648],
                        [   0.,    427.00564, 218.34591],
                        [   0.,      0.,    1.]])
        fx = camera_intrinsic[0, 0]
        fy = camera_intrinsic[1, 1]
        cx = camera_intrinsic[0, 2]
        cy = camera_intrinsic[1, 2]
        =====================================================
        :param cam_coord:
        :param f: [fx,fy]
        :param c: [cx,cy]
        :return:
        """
                 # Equivalent to: (f / dx) * (X / Z) = f * (X / Z) / dx
                 # Triangle transformation, / dx, + center_x
        u = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
        v = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
        d = cam_coord[..., 2]
        return u, v, d
 
    @staticmethod
    def convert_cc_to_ic(joint_cam):
        """
                 Camera coordinate system -> pixel coordinate system
        :param joint_cam:
        :return:
        """
                 # Camera coordinate system -> pixel coordinate system, and get relative depth
        # Subtract center depth
                 # Select the position of Pelvis pelvis as the center of the camera, use relative depth later
        root_idx = 0
        center_cam = joint_cam[root_idx]  # (x,y,z) mm
        joint_num = len(joint_cam)
        f = camera_intrinsic["f"]
        c = camera_intrinsic["c"]
                 # joint image, pixel coordinate system, Depth is relative depth mm
        joint_img = np.zeros((joint_num, 3))
        joint_img[:, 0], joint_img[:, 1], joint_img[:, 2] = Human36M.__cam2pixel(joint_cam, f, c)  # x,y
        joint_img[:, 2] = joint_img[:, 2] - center_cam[2]  # z
        return joint_img
 
 
if __name__ == "__main__":
    joint_world = [[-91.679, 154.404, 907.261],
                   [-223.23566, 163.80551, 890.5342],
                   [-188.4703, 14.077106, 475.1688],
                   [-261.84055, 186.55286, 61.438915],
                   [39.877888, 145.00247, 923.98785],
                   [-11.675994, 160.89919, 484.39148],
                   [-51.550297, 220.14624, 35.834396],
                   [-132.34781, 215.73018, 1128.8396],
                   [-97.1674, 202.34435, 1383.1466],
                   [-112.97073, 127.96946, 1477.4457],
                   [-120.03289, 190.96477, 1573.4],
                   [25.895456, 192.35947, 1296.1571],
                   [107.10581, 116.050285, 1040.5062],
                   [129.8381, -48.024918, 850.94806],
                   [-230.36955, 203.17923, 1311.9639],
                   [-315.40536, 164.55284, 1049.1747],
                   [-350.77136, 43.442127, 831.3473],
                   [-102.237045, 197.76935, 1304.0605]]
    joint_world = np.asarray(joint_world)
    kps_lines = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15),
                 (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
         # show in world coordinate system
    vis.vis_3d(joint_world, kps_lines, coordinate="WC", title="WC")
 
    human36m = Human36M()
 
         # show in camera coordinate system
    joint_cam = human36m.convert_wc_to_cc(joint_world)
    vis.vis_3d(joint_cam, kps_lines, coordinate="CC", title="CC")
    joint_img = human36m.convert_cc_to_ic(joint_cam)
 
         # show in pixel coordinate system
    kpt_2d = joint_img[:, 0:2]
    image_path = "E:\Pictures\20190408_211218.jpg"
    image = image_processing.read_image(image_path)
    image = image_processing.draw_key_point_in_image(image, key_points=[kpt_2d], pointline=kps_lines)
    image_processing.cv_show_image("image", image)
    '''
import numpy as np 

corners = np.array([[[1,2], 
					[3,4],
					[5,6],
					[7,8]],
					[[9, 10],
					[11,12],
					[13,14],
					[15,16]]], dtype=int)
print(len(corners))
