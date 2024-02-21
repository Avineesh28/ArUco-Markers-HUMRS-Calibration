#!/usr/bin/env python3
import matplotlib.pyplot as plt
from skimage import data, color, img_as_ubyte
from sensor_msgs.msg import Image, CameraInfo, Imu
import cv2
# import apriltag
import numpy as np
from scipy.spatial.transform import Rotation as R
import queue
from collections import deque
from mvextractor.videocap import VideoCap
from pupil_apriltags import Detector


import multiprocessing as mp
from multiprocessing import sharedctypes
import ctypes

def main():

    cap = cv2.VideoCapture('rtsp://admin:biorobotics!@192.168.1.64:554/Streaming/Channels/101')
    while True:
        ret, img = cap.read()
        height, width = img.shape[:2]
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Crop to Robot Specs -- Will need to remove for actual implementation
        # img = img[89:391, 53:587]   

        # print(height)
        # print(width)
        cv2.line(img,(960,0),(960,1080),(255,0,0),2)
        cv2.line(img,(0,540),(1920,540),(255,0,0),2)
        cv2.imshow("Camera Feed",img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
