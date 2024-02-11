#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Quaternion

from mvextractor.videocap import VideoCap
import multiprocessing as mp
from multiprocessing import sharedctypes

# Global Constants
cam = camera(rtsp_url="rtsp://admin:biorobotics!@192.168.1.64:554/Streaming/Channels/101")

# Calibration parameters
sideLength = 15  # Adjust based on your marker size (cm)
camera_mat = np.array([[1.51319600e3,0,1.02309616e+03],
                          [0,1.51474234e+03, 7.10080186e+02],
                          [0,0,1]])
dist_coeffs = np.array([-0.28857924,
                                0.22533772, 
                                0.00165789, 
                                0.00827434, 
                                -0.136742])
obj_points = np.array([[-sideLength / 2, sideLength / 2,0],
                           [sideLength / 2, sideLength / 2,0],
                           [sideLength / 2, -sideLength / 2,0],
                           [-sideLength / 2, -sideLength / 2,0]], dtype=np.float32)

# Publisher for estimated pose
pose_publisher = rospy.Publisher("/aruco_pose", PoseStamped, queue_size=1)

# Camera class 
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

    #used to grab frames from the cam connection process
    #[resize] param : % of size reduction or increase i.e 0.65 for 35% reduction  or 1.5 for a 50% increase
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

def aruco_pose_estimator():
    rospy.init_node("aruco_pose_estimator")
    # Camera setup  
    if not cam.cap.isOpened():
        rospy.logerr("Failed to open camera stream")
        return
    
    # Image subscriber
    image_sub = rospy.Subscriber("/humrs/camera/image", Image, image_callback)

def image_callback(msg):
  takeaction()    

def takeaction():
  try:
      while not rospy.is_shutdown():
        ret, frame = cam.get_frame()
        if not ret:
          continue

      # ArUco marker detection
      corners, ids, rejects = get_aruco_markers(frame)
      if any(ids):
      # Pose estimation for each detected marker
        for i in range(len(corners)):
          tvec, rvec = estimate_pose(corners[i], ids[i], camera_mat, dist_coeffs)
          if tvec is not None:
            try:
              Q=convert_rvec_to_quaternion(rvec, camera_mat, dist_coeffs)
              # Publish pose message
              publish_pose(pose_publisher, tvec,Q)
            except Exception as e:
              rospy.logerr(f"Error publishing pose for marker {ids[i]}: {e}")
      cv2.waitKey(1)
    
  except Exception as e:
        rospy.logerr(f"Error: {e}")
  finally:
      cam.release()
      cv2.destroyAllWindows()
      
def get_aruco_markers(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    return detector.detectMarkers(frame)

def estimate_pose(corner, id, camera_mat, dist_coeffs):
    try:
        _, rvec, tvec = cv2.solvePnP(obj_points, corner, camera_mat, dist_coeffs)
        return tvec, rvec
    except Exception as e:
        rospy.logwarn(f"Error estimating pose for marker {id}: {e}")
        return None, None

def convert_rvec_to_quaternion(rvec, camera_mat, dist_coeffs):
  #  Use cv2.solvePnP to get rotation matrix considering calibration
  _, rot_mat, _ = cv2.solvePnP(None, None, camera_mat, dist_coeffs, rvec, None)
  
  # Convert rotation matrix to quaternion
  quaternion = cv2.Rodrigues(rot_mat)[0].flatten()

  return quaternion

def publish_pose(publisher, tvec, quaternion):
  pose_msg = PoseStamped()
  pose_msg.header.stamp = rospy.Time.now()  # Add a timestamp
  pose_msg.header.frame_id = "camera_frame"  # Specify reference frame (adjust as needed)
  pose_msg.pose.position.x = tvec[0]  # Set position from tvec
  pose_msg.pose.position.y = tvec[1]
  pose_msg.pose.position.z = tvec[2]

  # Get the quaternion from rvec 
  pose_msg.pose.orientation.x = quaternion[0]  # Set orientation from quaternion
  pose_msg.pose.orientation.y = quaternion[1]
  pose_msg.pose.orientation.z = quaternion[2]
  pose_msg.pose.orientation.w = quaternion[3]

  # Publish the message
  publisher.publish(pose_msg)

if __name__ == "__main__":
    aruco_pose_estimator()
