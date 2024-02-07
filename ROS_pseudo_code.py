#!/usr/bin/env python3
import matplotlib.pyplot as plt
from skimage import data, color, img_as_ubyte
import rospy
from sensor_msgs.msg import Image, CameraInfo, Imu
import cv2
import numpy as np
import time
import rospkg
from complementary_filter import ComplementaryFilter
from nav_msgs.msg import Odometry
from sphere import R_btw_vecs
from circle_kf import CircleKF
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, QuaternionStamped, PointStamped
from std_msgs.msg import Bool, Header
import queue
from collections import deque
import pinocchio as pin
import skg

import multiprocessing as mp

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

class camera():
  def __init__(self,rtsp_url):        
    #load pipe for data transmittion to the process
    self.parent_conn, child_conn = mp.Pipe()
    #load process
    self.p = mp.Process(target=self.update, args=(child_conn,rtsp_url))        
    #start process
    self.p.daemon = True
    self.p.start()

  def end(self):
    #send closure request to process

    self.parent_conn.send(2)

  def update(self,conn,rtsp_url):
    #load cam into seperate process

    print("Cam Loading...")
    cap = cv2.VideoCapture(rtsp_url,cv2.CAP_FFMPEG)   
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Cam Loaded...")
    run = True

    while run:

      #grab frames from the buffer
      cap.grab()

      #recieve input data
      rec_dat = conn.recv()


      if rec_dat == 1:
        #if frame requested
        ret,frame = cap.read()
        conn.send(frame)

      elif rec_dat ==2:
        #if close requested
        cap.release()
        run = False

    print("Camera Connection Closed")        
    conn.close()

  def get_frame(self,resize=None):
    ###used to grab frames from the cam connection process

    ##[resize] param : % of size reduction or increase i.e 0.65 for 35% reduction  or 1.5 for a 50% increase

    #send request
    self.parent_conn.send(1)
    frame = self.parent_conn.recv()

    #reset request 
    self.parent_conn.send(0)

    #resize if needed
    if resize == None:            
      return frame
    else:
      return self.rescale_frame(frame,resize)

  def rescale_frame(self,frame, percent=65):
    return cv2.resize(frame,None,fx=percent,fy=percent) 

gravity = np.array([0., 0., -9.8])

do_undistortion = False

pipe_diameter = 0.202 # Pipe diameter underwater
# pipe_diameter = 0.13 # Diameter of circle I printed on paper (also outer diameter of thicker circle)
# pipe_diameter = 0.12 # Diameter of thicker circle I printed on paper

orange_hsv_low = (160, 100, 100)
orange_hsv_high = (180, 255, 255)

# Thresholds for the orange circle I printed on paper
# orange_hsv_low = (0, 100, 150)
# orange_hsv_high = (10, 200, 255)

saved = False

rospack = rospkg.RosPack()
path = rospack.get_path('humrs_control')
urdf_file = path + '/../urdf/snake.urdf'

dist_coeffs = None
new_camera_matrix = None
roi = None

img_step = 5760

img_stamp = 0

use_hough = False

prev_rad = None
prev_center = None

def locate_pipe_in_image(bgr):
  global camera_matrix
  global dist_coeffs
  global new_camera_matrix
  
  # This is slow, take out if it's not necessary
  if do_undistortion:
    bgr = cv2.undistort(bgr, camera_matrix, dist_coeffs, None, new_camera_matrix)
  else:
    bgr = np.copy(bgr)
    
  rmat180 = cv2.getRotationMatrix2D((bgr.shape[1] / 2, bgr.shape[0] / 2), 180, 1)

  hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

  thresh = cv2.inRange(hsv, orange_hsv_low, orange_hsv_high)

  '''
  global saved
  if not saved:
    cv2.imwrite(path + '/orange_pipe.png', bgr)
    saved = True
  '''

  thresh_copy = cv2.warpAffine(thresh, rmat180, (thresh.shape[1], thresh.shape[0]))

  # Original image is 1920 x 1080
  thresh = cv2.medianBlur(thresh, 5)
  scale_fact = 8
  thresh = cv2.resize(thresh, (thresh.shape[1]//scale_fact, thresh.shape[0]//scale_fact))

  rows = thresh.shape[0]

  if use_hough:
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, rows / 2,
                               param1=100, param2=20,
                               minRadius=0, maxRadius=0)
    
    best_rad = -1
    best_center = (0, 0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0]*scale_fact, i[1]*scale_fact)
            radius = i[2]*scale_fact
            if radius > best_rad:
              best_rad = radius
              best_center = (center[0], center[1])
  else:
    nonzero_idx = np.flip(np.array(np.nonzero(thresh)).transpose(), 1)
    no_pixels = len(nonzero_idx) == 0

    if not no_pixels:
      mean = np.mean(nonzero_idx, 0)
      std = np.std(nonzero_idx, 0)
      nonzero_idx = nonzero_idx[np.all(np.abs(nonzero_idx - mean) < 3*std, 1)]

    no_pixels = len(nonzero_idx) == 0

    if no_pixels:
      best_rad = -1
      best_center = (np.nan, np.nan)
      prev_rad = None
    else:
      chull = cv2.convexHull(nonzero_idx.astype(np.float32)).squeeze(1)

      split_chull = [[]]
      do_split = False
      for point in chull:
        if point[0] == 0 or point[0] == thresh.shape[1] - 1 or point[1] == 0 or point[1] == thresh.shape[0] - 1:
          do_split = True
        elif do_split:
          do_split = False
          split_chull.append([])
          split_chull[-1].append(point)
        else:
          split_chull[-1].append(point)

      best_contour_idx = np.argmax([len(contour) for contour in split_chull])
      best_contour = split_chull[best_contour_idx]

      if len(chull) < 10:
        best_rad = -1
        best_center = (np.nan, np.nan)
        prev_rad = None
      else:
        '''
        bgr_scaled = cv2.resize(bgr, (bgr.shape[1]//scale_fact, bgr.shape[0]//scale_fact))
        for i, point in enumerate(best_contour[:-1]):
          cv2.line(bgr_scaled, (int(point[0]), int(point[1])), (int(best_contour[i + 1][0]), int(best_contour[i + 1][1])), [0, 255, 0], 2)
        cv2.imshow('bgr_scaled', bgr_scaled)
        cv2.waitKey(1)
        cv2.imshow('thresh', thresh)
        cv2.waitKey(1)
        '''

        best_rad, best_center = skg.nsphere_fit(best_contour, 1)
        best_rad = int(best_rad*scale_fact)
        best_center = (int(best_center[0]*scale_fact), int(best_center[1]*scale_fact))

  circle_diameter = 2*best_rad

  if best_rad > 0:
    cv2.circle(bgr, best_center, best_rad, (255, 0, 255), 20)

  bgr_annotated = cv2.warpAffine(bgr, rmat180, (bgr.shape[1], bgr.shape[0]))

  return best_center, circle_diameter, bgr_annotated, thresh_copy

def locate_pipe_in_image_avg(bgr):
  global camera_matrix
  global dist_coeffs
  global new_camera_matrix

  # This is slow, take out if it's not necessary
  if do_undistortion:
    bgr = cv2.undistort(bgr, camera_matrix, dist_coeffs, None, new_camera_matrix)

  rmat180 = cv2.getRotationMatrix2D((bgr.shape[1] / 2, bgr.shape[0] / 2), 180, 1)

  hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

  thresh = cv2.inRange(hsv, orange_hsv_low, orange_hsv_high)

  thresh_copy = cv2.warpAffine(thresh, rmat180, (thresh.shape[1], thresh.shape[0]))

  M = cv2.moments(thresh)

  area = np.sum(thresh)/255
 
  if area > 0:
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    circle_center = (cX, cY)
    cv2.circle(bgr, circle_center, 20, (255, 0, 255), 20)
  else:
    cX = np.nan
    cY = np.nan
    circle_center = (cX, cY)

  bgr_annotated = cv2.warpAffine(bgr, rmat180, (bgr.shape[1], bgr.shape[0]))

  return circle_center, area, bgr_annotated, thresh_copy

bgr_copy = None

def camera_callback(msg):
  global bgr_copy
  global img_step
  global img_stamp

  bgr_copy = np.copy(np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1))
  img_step = msg.step
  img_stamp = msg.header.stamp.to_sec()
  
imu_msg_queue = queue.Queue()
def imu_callback(msg):
  global imu_msg_queue
  imu_msg_queue.put(msg)

info_msg = None
def info_callback(msg):
  global info_msg
  info_msg = msg

imu_odom_queue = queue.Queue()
def imu_odom_callback(msg):
  global imu_odom_queue
  imu_odom_queue.put(msg)

rospy.init_node('circle_detection_node', anonymous=True)
# image_sub = rospy.Subscriber('/humrs/camera/image', Image, camera_callback, queue_size=1, buff_size=6220800*2) # I don't think setting queue_size and buff_size helped with anything
# info_sub = rospy.Subscriber('/humrs/camera/camera_info', CameraInfo, info_callback)

vision_processing_delay_estimate = rospy.get_param('vision_processing_delay_estimate')
vision_ethernet_delay_estimate = rospy.get_param('vision_ethernet_delay_estimate')
delay_estimate = vision_processing_delay_estimate + vision_ethernet_delay_estimate

# Get camera info
fx = rospy.get_param("/fx")
fy = rospy.get_param("/fy")
cx = rospy.get_param("/cx")
cy = rospy.get_param("/cy")
dist = rospy.get_param("/dist")

cam = camera(('rtsp://admin:biorobotics!@192.168.1.64:554/Streaming/Channels/101'))

print('Camera is alive?: %d' %(cam.p.is_alive()))

got_frame = False
print('Getting first camera frame')
frame = cam.get_frame(1)
print('Got first camera frame')
camera_info = CameraInfo()
camera_info.header = Header()
camera_info.header.frame_id = 'camera'
camera_info.width = frame.shape[1]
camera_info.height = frame.shape[0]
camera_info.distortion_model = 'plumb_bob'
camera_info.K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
camera_info.D = dist
camera_info.R = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
camera_info.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1.0, 0]

imu_name = 'radial1__IMU_1'
imu_sub = rospy.Subscriber('/humrs/fbk/' + imu_name, Imu, imu_callback)

imu_odom_sub = rospy.Subscriber('/humrs/imu_odom', Odometry, imu_odom_callback)

delayed_odom_pub = rospy.Publisher('/humrs/delayed_odom', Odometry, queue_size=1)
delayed_odom = Odometry()
delayed_odom.header.frame_id = 'world'
delayed_odom.child_frame_id = 'head'

thresh_pub = rospy.Publisher('/humrs/circle_thresh', Image, queue_size=1)
thresh_msg = Image()
thresh_msg.encoding = 'mono8'
thresh_msg.header.frame_id = 'camera'
thresh_msg.is_bigendian = 0

circle_annotation_pub = rospy.Publisher('/humrs/circle_annotation', Image, queue_size=1)
circle_annotation_msg = Image()
circle_annotation_msg.encoding = 'bgr8'
circle_annotation_msg.header.frame_id = 'camera'
circle_annotation_msg.is_bigendian = 0

image_pub = rospy.Publisher('/humrs/camera/image', Image, queue_size=1)
image_msg = Image()
image_msg.encoding = 'bgr8'
image_msg.header.frame_id = 'camera'
image_msg.is_bigendian = 0

info_pub = rospy.Publisher('/humrs/camera/camera_info', CameraInfo, queue_size=1)

pipe_pub = rospy.Publisher('/humrs/pipe', Marker, queue_size=1)
pipe_marker = Marker()
pipe_marker.header.frame_id = 'world'
pipe_marker.ns = 'estimation_node'
pipe_marker.action = Marker.ADD
pipe_marker.pose.position.x = 1.5
pipe_marker.pose.position.y = 0
pipe_marker.pose.position.z = 0
quat = R.from_rotvec([0., np.pi/2, 0.]).as_quat()
pipe_marker.pose.orientation.x = quat[0]
pipe_marker.pose.orientation.y = quat[1]
pipe_marker.pose.orientation.z = quat[2]
pipe_marker.pose.orientation.w = quat[3]

pipe_marker.type = Marker.CYLINDER
pipe_marker.scale.x = 0.202
pipe_marker.scale.y = 0.202
pipe_marker.scale.z = 3

color = [0.5, 0.5, 0.5]
pipe_marker.color.b = color[0]
pipe_marker.color.g = color[1]
pipe_marker.color.r = color[2]
pipe_marker.color.a = 0.5

tape_pub = rospy.Publisher('/humrs/tape', Marker, queue_size=1)
tape_marker = Marker()
tape_marker.type = Marker.LINE_STRIP
tape_marker.action = Marker.ADD
tape_marker.pose.position.x = 0
tape_marker.pose.position.y = 0
tape_marker.pose.position.z = 0
tape_marker.pose.orientation.w = 1
tape_marker.pose.orientation.x = 0
tape_marker.pose.orientation.y = 0
tape_marker.pose.orientation.z = 0
tape_marker.points = [Point(0.0, 0.101*np.cos(theta), 0.101*np.sin(theta)) for theta in np.linspace(0, 2*np.pi)]
tape_marker.scale.x = 0.05
tape_marker.color.r = 245/255
tape_marker.color.g = 96/255
tape_marker.color.b = 66/255
tape_marker.color.a = 1
tape_marker.header.frame_id = "world"
tape_marker.ns = "estimation_node"

pos_init_pub = rospy.Publisher('/humrs/position_initialized', Bool, queue_size=1)

circle_pub = rospy.Publisher('/humrs/circle', PointStamped, queue_size=1)
circle_msg = PointStamped()
circle_msg.header.frame_id = 'world'

dt = 0.02
rate = rospy.Rate(1/dt)

while imu_msg_queue.empty() or imu_odom_queue.empty() is None:
  rate.sleep()

# Sleep and let the IMU and odometry queues fill up
rospy.sleep(delay_estimate)

camera_matrix = np.reshape(camera_info.K, (3, 3))
dist_coeffs = np.array([camera_info.D])

# While testing in air
'''
fx = 1.42939882e+03
fy = 1.42371371e+03
cx = 9.17423917e+02
cy = 5.11350787e+02
camera_matrix = np.eye(3)
camera_matrix[0, 0] = fx
camera_matrix[1, 1] = fy
camera_matrix[0, 2] = cx
camera_matrix[1, 2] = cy
dist_coeffs = np.array([[-4.92746521e-01, 7.58441397e-01, -1.88877950e-03, 6.39318378e-04, -1.29355686e+00]])
'''

if do_undistortion:
  new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (info_msg.width, info_msg.height), 0, (info_msg.width, info_msg.height))
else:
  new_camera_matrix = camera_matrix

kf = CircleKF(urdf_file, new_camera_matrix, imu_name, pipe_diameter)
prev_time = rospy.Time.now().to_sec()

do_global_position_kf = True

start_time = rospy.Time.now().to_sec()

prev_imu_odom = None

def on_shutdown():
  cam.end()
  cv2.destroyAllWindows()
rospy.on_shutdown(on_shutdown)

while not rospy.is_shutdown():
  get_frame_start = rospy.Time.now().to_sec()

  bgr_copy = cam.get_frame(1)

  image_msg.header.stamp = rospy.Time.now()
  image_msg.data = bgr_copy.tobytes()
  image_msg.width = bgr_copy.shape[1]
  image_msg.height = bgr_copy.shape[0]
  image_msg.step = img_step
  image_pub.publish(image_msg)

  camera_info.header.stamp = rospy.Time.now()
  info_pub.publish(camera_info)

  get_frame_end = rospy.Time.now().to_sec()
  # print('Frame getting time: %f' %(get_frame_end - get_frame_start))

  # print('Latest image stamp minus current time: %f' %(img_stamp - rospy.Time.now().to_sec()))
  if do_global_position_kf:
    circle_center, circle_diameter, bgr_annotated, thresh_copy = locate_pipe_in_image(bgr_copy)
  else:
    circle_center, circle_diameter, bgr_annotated, thresh_copy = locate_pipe_in_image_avg(bgr_copy)

  cur_time = rospy.Time.now().to_sec()
  dt = cur_time - prev_time
  prev_time = cur_time

  # Look through the IMU and odometry queues to figure out which messages correspond to the current image
  while True:
    imu_msg = imu_msg_queue.get()
    if imu_msg.header.stamp.to_sec() > cur_time - delay_estimate:
      break

  while True:
    imu_odom = imu_odom_queue.get()
    if imu_odom.header.stamp.to_sec() > cur_time - delay_estimate:
      break

  imu_a = np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])
  imu_w = np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z])

  head_quat = np.array([imu_odom.pose.pose.orientation.x, imu_odom.pose.pose.orientation.y, imu_odom.pose.pose.orientation.z, imu_odom.pose.pose.orientation.w])

  '''
  print(kf.rmat_imu_head)
  grav_local = kf.rmat_imu_head.transpose()@R.from_quat(head_quat).inv().apply(np.array([0., 0., 9.8]))
  print(imu_a - grav_local)
  quit()
  '''

  head_rmat = R.from_quat(head_quat).as_matrix()

  head_position_estimate = np.zeros(3)

  if do_global_position_kf:
    if kf.initialized:
      pos = np.array([imu_odom.pose.pose.position.x, imu_odom.pose.pose.position.y, imu_odom.pose.pose.position.z])
      pos_prev = np.array([prev_imu_odom.pose.pose.position.x, prev_imu_odom.pose.pose.position.y, prev_imu_odom.pose.pose.position.z])
      delta_pos = pos - pos_prev

      vel = np.array([imu_odom.twist.twist.linear.x, imu_odom.twist.twist.linear.y, imu_odom.twist.twist.linear.z])
      vel_prev = np.array([prev_imu_odom.twist.twist.linear.x, prev_imu_odom.twist.twist.linear.y, prev_imu_odom.twist.twist.linear.z])
      delta_v = vel - vel_prev

      delta_t = imu_odom.header.stamp.to_sec() - prev_imu_odom.header.stamp.to_sec()

      kf.predict_from_deltas(delta_pos, delta_v, vel_prev, delta_t)

      if circle_diameter > 0:
        kf.correct(circle_center, circle_diameter, head_rmat, True)
    elif circle_diameter > 0:
      kf.initialize(circle_center, circle_diameter, head_rmat)

    head_position_estimate = np.copy(kf.kf_x[:3])
  else:
    if circle_diameter > 0:
      t_head_circle = kf.get_t_head_circle(circle_center, circle_diameter)

      head_position_estimate[0] = -t_head_circle[0]
      head_position_estimate[1] = -t_head_circle[1]
      head_position_estimate[2] = t_head_circle[2]

  prev_imu_odom = imu_odom

  delayed_odom.pose.pose.position.x = head_position_estimate[0]
  delayed_odom.pose.pose.position.y = head_position_estimate[1]
  delayed_odom.pose.pose.position.z = head_position_estimate[2]

  delayed_odom.pose.pose.orientation.x = imu_odom.pose.pose.orientation.x
  delayed_odom.pose.pose.orientation.y = imu_odom.pose.pose.orientation.y
  delayed_odom.pose.pose.orientation.z = imu_odom.pose.pose.orientation.z
  delayed_odom.pose.pose.orientation.w = imu_odom.pose.pose.orientation.w
  delayed_odom.twist.twist.linear.x = kf.kf_x[3]
  delayed_odom.twist.twist.linear.y = kf.kf_x[4]
  delayed_odom.twist.twist.linear.z = kf.kf_x[5]
  delayed_odom.twist.twist.angular.x = imu_odom.twist.twist.angular.x
  delayed_odom.twist.twist.angular.y = imu_odom.twist.twist.angular.y
  delayed_odom.twist.twist.angular.z = imu_odom.twist.twist.angular.z
  delayed_odom.header.stamp = rospy.Time.now()
  delayed_odom_pub.publish(delayed_odom)

  if thresh_copy is not None:
    thresh_msg.header.stamp = rospy.Time.now()
    thresh_msg.data = thresh_copy.tobytes()
    thresh_msg.width = thresh_copy.shape[1]
    thresh_msg.height = thresh_copy.shape[0]
    thresh_msg.step = 8450
    thresh_pub.publish(thresh_msg)

  if bgr_annotated is not None:
    circle_annotation_msg.header.stamp = rospy.Time.now()
    circle_annotation_msg.data = bgr_annotated.tobytes()
    circle_annotation_msg.width = bgr_annotated.shape[1]
    circle_annotation_msg.height = bgr_annotated.shape[0]
    circle_annotation_msg.step = img_step
    circle_annotation_pub.publish(circle_annotation_msg)

  pipe_marker.header.stamp = rospy.Time.now()
  pipe_pub.publish(pipe_marker)
  tape_marker.header.stamp = rospy.Time.now()
  tape_pub.publish(tape_marker)

  normalized_circle_center = np.array([circle_center[0], circle_center[1], 1])
  normalized_circle_center = np.linalg.solve(new_camera_matrix, normalized_circle_center)
  circle_msg.point.x = normalized_circle_center[0]
  circle_msg.point.y = normalized_circle_center[1]
  circle_msg.point.z = circle_diameter
  circle_msg.header.stamp = rospy.Time.now()
  circle_pub.publish(circle_msg)

  pos_init_pub.publish(Bool(kf.initialized))

  rate.sleep()