#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2  
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np 
from yolov8_ros_msgs.msg import BoundingBox, BoundingBoxes
import message_filters 
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
import message_filters
import time

#-------------------------相机内参-----------------------------#
left_camera_matrix = np.array([[843.98893,0., 471.48569],[0., 844.10121, 363.66004],[0.,0.,1.]])
right_camera_matrix = np.array([[834.33816,0., 480.58831],[0., 834.67783, 357.65044],[0.,0.,1.]])
left_distortion = np.array([[0.008840, -0.007895, 0.001425, -0.001180, 0.000000]])
right_distortion = np.array([[0.003516, -0.007326, 0.000058, -0.000106, 0.000000]])
R = np.array([[0.9999582260830783, -0.00025211415115678273, -0.00913687732424046],
              [0.0003125877185250819, 0.9999780532538496, 0.006617801716546528],
              [0.009135008358050927, -0.0066203813406846435, 0.9999363590615167]])
T = np.array([-0.3022654121044337, 0.000630765386765559, -0.018435733311510077])
size = (960, 720)
baseline = 0.3 
focal_length = 843.98893
#使用cv2.stereoRectify()函数计算校正变换矩阵，输入参数：相机内参；输出参数：R1、R2（矫正后的左右相机的旋转矩阵）、P1、P2（矫正后的左右相机的投影矩阵）、Q（用于深度计算的重投影矩阵）、validPixROI1、validPixROI2（矫正后的有效像素区域）
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
# cv2.initUndistortRectifyMap()函数用于生成两个映射矩阵，这些矩阵可以用于将原始畸变图像转换为矫正后的图像
# left_map1,left_map2:两个映射矩阵，分别表示x和y方向上的映射关系
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

rectify_left_image = None
rectify_right_image = None

#-------------------------------------------------------------#
class image_converter:
    def __init__(self):    
        # 创建cv_bridge，声明图像订阅者
        self.bridgel = CvBridge()
        self.bridger = CvBridge()
        self.image_sub_left = message_filters.Subscriber("/airsim_node/drone_1/front_left/Scene", Image)
        self.image_sub_right = message_filters.Subscriber("/airsim_node/drone_1/front_right/Scene", Image)
        self.yolo_sub = message_filters.Subscriber("/yolov8/BoundingBoxes", BoundingBoxes)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub_left, self.image_sub_right,self.yolo_sub], 10, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.image_callback)
        self.target_point_pub = rospy.Publisher("/target_points", PointStamped, queue_size=1)
    def image_callback(self,left_image_msg,right_image_msg,yolo_msg):
        try:
            # Convert the ROS image messages to OpenCV images in BGR8 format
            left_image = self.bridgel.imgmsg_to_cv2(left_image_msg, "bgr8")
            right_image = self.bridger.imgmsg_to_cv2(right_image_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Convert the left and right images to grayscale
        gray_left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # Rectify the grayscale images using the precomputed maps
        rectify_left_image = cv2.remap(gray_left_image, left_map1, left_map2, cv2.INTER_LINEAR)
        rectify_right_image = cv2.remap(gray_right_image, right_map1, right_map2, cv2.INTER_LINEAR)

        # Convert the rectified images back to BGR format for display
        imageL = cv2.cvtColor(rectify_left_image, cv2.COLOR_GRAY2BGR)
        imageR = cv2.cvtColor(rectify_right_image, cv2.COLOR_GRAY2BGR)
        # 显示图像
        # cv2.imshow("left_image", imageL)
        # cv2.imshow("right_image", imageR)
        # cv2.waitKey(3)

        # 计算视差图代替深度图
        depth_map = sgbm(rectify_left_image, rectify_right_image,Q)
        depth_map = depth_map.astype(np.float32)#类型转换
        # cv2.imshow("depth_map", depth_map)
        # cv2.waitKey(3)

        # 计算方框中心点
        center_y, center_x = target_points_get(yolo_msg)
        print(center_y,center_x)

        # 检查 center_y 和 center_x 是否为 None
        while center_y is None or center_x is None:
            print("Waiting for center_y and center_x to be refreshed...")
            time.sleep(1)  # 等待一段时间再重试
            center_y, center_x = target_points_get(yolo_msg)

        # 计算目标点在相机坐标系下的坐标
        threeD = cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=True)
        threeD = threeD * 16
        threeD = threeD.astype(np.float32)
        camera_x, camera_y, camera_z = threeD[center_y, center_x]
        print(camera_x, camera_y, camera_z)
        # 创建 PointStamped 消息
        point_stamped = PointStamped()
        point_stamped.header = yolo_msg.header  # 使用接收到的消息的 header
        point_stamped.point.x = camera_x
        point_stamped.point.y = camera_y
        point_stamped.point.z = camera_z
        self.target_point_pub.publish(point_stamped)    


def sgbm(rectify_left_image, rectify_right_image,Q):
   blockSize = 3
   img_channels = 3
   stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=64,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
   #计算视差图
   disparity = stereo.compute(rectify_left_image, rectify_right_image)
   disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
   #颜色映射
   dis_color = disparity
   dis_color = cv2.normalize(dis_color, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
   dis_color = cv2.applyColorMap(dis_color, 2)
   cv2.imshow("dis_color", dis_color)
   cv2.waitKey(3)
   
   #计算深度图
   depth_map = (baseline * focal_length )/ (disparity+1e-10) #防止除0
   depth_map = depth_map.astype(np.float32)#类型转换
   return disparity

def target_points_get (bounding_boxes_msg):
    center_x = None
    center_y = None
    for bounding_box in bounding_boxes_msg.bounding_boxes:
        xmin = bounding_box.xmin
        xmax = bounding_box.xmax
        ymin = bounding_box.ymin
        ymax = bounding_box.ymax
        #用于判断是否成功
        print(f"Box: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")

        #计算中心点
        center_x = int((xmin + xmax) // 2)
        center_y = int((ymin + ymax) // 2)
    return center_y,center_x

    


if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("cv_bridge_test")
        rospy.loginfo("Starting cv_bridge_test node")
        image_converter()
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down cv_bridge_test node.")
        cv2.destroyAllWindows()
