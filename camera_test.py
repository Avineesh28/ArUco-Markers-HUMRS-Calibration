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

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

class camera():
  def __init__(self,rtsp_url):        
    #load pipe for data transmittion to the process
    self.parent_conn, child_conn = mp.Pipe()
    #figure out how big the frames coming in will be
    cap = VideoCap()
    cap.open(rtsp_url)   
    ret = False
    while not ret:
      cap.grab()
      ret,frame,_,_,timestamp=cap.retrieve()
    cap.release()
    self.frame_shape=frame.shape
    self.frame_dtype=frame.dtype
    self.shared_array = sharedctypes.RawArray(np.ctypeslib.as_ctypes_type(self.frame_dtype), int(np.prod(self.frame_shape)))
    self.frame_buffer=np.frombuffer(self.shared_array, dtype=self.frame_dtype).reshape(self.frame_shape)
    #load process
    self.p = mp.Process(target=self.update, args=(child_conn,rtsp_url,self.shared_array,self.frame_dtype,self.frame_shape))        
    #start process
    self.p.daemon = True
    self.p.start()

  def end(self):
    #send closure request to process

    self.parent_conn.send((2,None))

  def update(self,conn,rtsp_url,shared_array,frame_dtype,frame_shape):
    #load cam into seperate process
    buffer=np.frombuffer(shared_array, dtype=frame_dtype).reshape(frame_shape)
    print("Cam Loading...")
    cap = VideoCap()
    cap.open(rtsp_url)   
    print("Cam Loaded...")
    run = True
    cap.grab()
    ret,frame,_,_,timestamp=cap.retrieve()
    while run:

      #recieve input data
      if conn.poll():
        rec_dat = conn.recv()
      else:
        rec_dat=3


      if rec_dat == 1:
        #if frame requested
        while not ret:
          cap.grab()
          ret,frame,_,_,timestamp=cap.retrieve()
        np.copyto(buffer,frame)
        conn.send((timestamp))
        ret=False

      elif rec_dat ==2:
        #if close requested
        cap.release()
        run = False
      else:
        cap.grab()
        ret,frame,_,_,timestamp=cap.retrieve()
    print("Camera Connection Closed")        
    conn.close()

  def get_frame(self,resize=None):
    ###used to grab frames from the cam connection process

    ##[resize] param : % of size reduction or increase i.e 0.65 for 35% reduction  or 1.5 for a 50% increase

    #send request
    self.parent_conn.send(1)
    timestamp = self.parent_conn.recv()

    #reset request
    self.parent_conn.send(0)
    #resize if needed
    if resize == None:            
      return timestamp,self.frame_buffer
    else:
      return timestamp,self.rescale_frame(self.frame_buffer,resize)

  def rescale_frame(self,frame, percent=65):
    return cv2.resize(frame,None,fx=percent,fy=percent) 


cam = camera(('rtsp://admin:biorobotics!@192.168.1.64:554/Streaming/Channels/101'))

print('Camera is alive?: %d' %(cam.p.is_alive()))

got_frame = False
print('Getting first camera frame')
first_timestamp,frame = cam.get_frame(1)
print('Got first camera frame')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray)

at_detector = Detector(
  families="tag36h11",
  nthreads=1,
  quad_decimate=1.0,
  quad_sigma=0.0,
  refine_edges=1,
  decode_sharpening=0.25,
  debug=0
)

fx = 1.42939882e+03
fy = 1.42371371e+03
cx = 9.17423917e+02
cy = 5.11350787e+02
rot_rec=[]
pos_rec=[]
cam_parms=[fx,fy,cx,cy]
timestamp,frame = cam.get_frame(1)
while timestamp-first_timestamp<10:
  timestamp,frame = cam.get_frame(1)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # print(first_timestamp)
  # print(timestamp-first_timestamp)
  img_out=at_detector.detect(gray,estimate_tag_pose=True,camera_params=cam_parms,tag_size=81.28/1000.)
  if len(img_out)>0:
    for j in img_out:
      if j.tag_id==580:
        rotm=R.from_matrix(j.pose_R)
        rot_rec.append(rotm.as_rotvec())
        pos_rec.append(j.pose_t)
    # print(img_out[1].tag_id)
    
    # if img_out[1].tag_id==580:
      #  rotm=R.from_matrix(img_out[1].pose_R)
      #  rot_rec.append(rotm[1].as_rotvec())
      #  pos_rec.append(pos_rec)

rot_rec=np.array(rot_rec)
pos_rec=np.array(pos_rec)
print("Print out the mean and standard deviation of position")
for i in range(3):
  print(np.mean(pos_rec[:,i,0]),np.std(pos_rec[:,i,0]))
  
print("Print out the mean and standard deviation of rotation")
for i in range(3):
  print(np.mean(rot_rec[:,i]),np.std(rot_rec[:,i]))



# with open('RotOutput.txt', 'w') as myFile:
#   for i in rot_rec:
#     myFile.write(str(i) + "\n")

# with open('PosOutput.txt', 'w') as myFile:
#   for i in pos_rec:
#     myFile.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]))
#     myFile.write("\n")
    

# print(np.shape(rot_rec))
# for i in rot_rec:
#   print(i)
  
# print(np.shape(pos_rec))
# for i in pos_rec:
#   print(np.array(i).transpose())
    # print(img_out)
  # try:
  #   print(at_detector.tag_id)
  # except:
  #   pass
# print(frame)
# print('Got first camera frame')

# timestamp,bgr_copy = cam.get_frame(1)
# options = apriltag.DetectorOptions(families="tag36h11")
# detector = apriltag.Detector()
# results = detector.detect(gray)