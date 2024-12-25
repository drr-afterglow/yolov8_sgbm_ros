#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image

class ImageRepublisher:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('image_republisher', anonymous=True)

        # 订阅原始的左右相机话题
        self.left_image_sub = rospy.Subscriber('/airsim_node/drone_1/front_left/Scene', Image, self.left_image_callback)
        self.right_image_sub = rospy.Subscriber('/airsim_node/drone_1/front_right/Scene', Image, self.right_image_callback)

        # 创建新的话题发布者
        self.left_image_pub = rospy.Publisher('/airsim_node/drone_1/front_left/Scene_1', Image, queue_size=10)
        self.right_image_pub = rospy.Publisher('/airsim_node/drone_1/front_right/Scene_1', Image, queue_size=10)

        # 初始化图像消息
        self.left_image_msg = None
        self.right_image_msg = None

        # 设置发布频率为16Hz
        self.rate = rospy.Rate(17.5)

    def left_image_callback(self, msg):
        self.left_image_msg = msg

    def right_image_callback(self, msg):
        self.right_image_msg = msg

    def republish_images(self):
        while not rospy.is_shutdown():
            if self.left_image_msg is not None and self.right_image_msg is not None:
                # 发布左右相机图像
                self.left_image_pub.publish(self.left_image_msg)
                self.right_image_pub.publish(self.right_image_msg)

            # 按照16Hz的频率休眠
            self.rate.sleep()

if __name__ == '__main__':
    try:
        image_republisher = ImageRepublisher()
        image_republisher.republish_images()
    except rospy.ROSInterruptException:
        pass

