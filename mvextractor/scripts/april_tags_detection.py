#!/usr/bin/python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Quaternion
from mvextractor.videocap import VideoCap
import multiprocessing as mp
from collections import defaultdict
from multiprocessing import sharedctypes

# Global Constants
store=defaultdict(list)
store["loop_times"]=[]

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

# np.set_printoptions(linewidth=np.inf)
# np.set_printoptions(suppress=True)

class Camera():
  
  # def __init__(self,rtsp_url):        
  def __init__(self):  
    
    #load pipe for data transmittion to the process
    self.parent_conn, child_conn = mp.Pipe()
    
    #figure out how big the frames coming in will be
    cap = VideoCap(0)
    cap.open("0")   
    ret = False
    
    rospy.loginfo("Reached")

    while not ret:
      cap.grab()
      ret,frame,_,_,timestamp=cap.retrieve()

    rospy.loginfo("Reached")
    
    cap.release()
    
    self.frame_shape=frame.shape
    self.frame_dtype=frame.dtype
    self.shared_array = sharedctypes.RawArray(np.ctypeslib.as_ctypes_type(self.frame_dtype), int(np.prod(self.frame_shape)))
    self.frame_buffer=np.frombuffer(self.shared_array, dtype=self.frame_dtype).reshape(self.frame_shape)
    
    #load process
    self.p = mp.Process(target=self.update, args=(child_conn,self.shared_array,self.frame_dtype,self.frame_shape))        
    
    #start process
    self.p.daemon = True
    self.p.start()

  def end(self):
    
    #send closure request to process
    self.parent_conn.send((2,None))

  def update(self,conn,shared_array,frame_dtype,frame_shape):
    
    #load cam into seperate process
    buffer=np.frombuffer(shared_array, dtype=frame_dtype).reshape(frame_shape)
    
    print("Cam Loading...")
    cap = VideoCap(0)
    cap.open(0)   
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

def april_detector():
    
    rospy.init_node("april_detector")
    cam = Camera()
    
    # Camera setup  
    if not cam.cap.isOpened():
        rospy.logerr("Failed to open camera stream")
        return
    
    # Image subscriber
    image_sub = rospy.Subscriber("/humrs/camera/image", Image, camera_callback, queue_size=1)
    
    image_pub = rospy.Publisher("/humrs/camera/image", Image, queue_size=1)
    
    image_msg = Image()
    image_msg.encoding = 'bgr8'
    image_msg.header.frame_id = 'camera'
    image_msg.is_bigendian = 0
    
    print('Camera is alive?: %d' %(cam.p.is_alive()))
    got_frame = False

    print('Getting first camera frame')
    first_timestamp,frame = cam.get_frame(1)

    print('Got first camera frame')

    camera_info = CameraInfo()
    camera_info.header = Header()
    camera_info.header.frame_id = 'camera'
    camera_info.width = frame.shape[1]
    camera_info.height = frame.shape[0]
    camera_info.distortion_model = 'plumb_bob'

    last_loop=None

    while not rospy.is_shutdown():

      get_frame_start = rospy.Time.now().to_sec()

      if last_loop is not None:
        store["loop_times"].append(get_frame_start-last_loop)

      last_loop=get_frame_start
      timestamp,bgr_copy = cam.get_frame(1)
      timestamp=rospy.Time.from_sec(timestamp)
      t=print_dur("get frame",get_frame_start,store,False)

      image_msg.header.stamp = timestamp
      image_msg.data = bgr_copy.tobytes()
      image_msg.width = bgr_copy.shape[1]
      image_msg.height = bgr_copy.shape[0]
      image_msg.step = img_step
      image_pub.publish(image_msg)
      t=print_dur("publish image",t,store)
      
    rospy.on_shutdown(on_shutdown(cam))

def camera_callback(msg):
  
  global bgr_copy
  global img_step
  global img_stamp

  bgr_copy = np.copy(np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1))
  img_step = msg.step
  img_stamp = msg.header.stamp.to_sec()

def on_shutdown(cam):
        cam.end()
        cv2.destroyAllWindows()
        for key in store:
          print(key+": "+str(np.mean(store[key])))

def print_dur(phase,t_last,store,skip=False):
  
  now=rospy.Time.now().to_sec()

  if not skip:
    store[phase].append(now-t_last)
    #print(phase+": "+str(now-t_last))

  return now

if __name__ == "__main__":
    april_detector()
