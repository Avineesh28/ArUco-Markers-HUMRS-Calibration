#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2

class Camera:
    def __init__(self):
        rospy.init_node('camera',anonymous=True)
        self.vid=cv2.VideoCapture(0)

    def main_control(self):
        while(True):
            re, frame =self.vid.read()
            cv2.imshow('frame',frame)
            image_pub = rospy.Publisher('/camera_feed', Image, queue_size=1)
            image_msg = Image()
            image_msg.encoding = 'bgr8'
            image_msg.header.frame_id = 'camera'
            image_msg.is_bigendian = 0
            image_msg.data = frame.tobytes()
            image_msg.width = frame.shape[1]
            image_msg.height = frame.shape[0]
            image_pub.publish(image_msg)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.vid.release()
        cv2.destroyAllWindows()
if __name__ =='__main__':
    cam=Camera()
    r=rospy.Rate(60)
    while not rospy.is_shutdown():
        cam.main_control()
        r.sleep()