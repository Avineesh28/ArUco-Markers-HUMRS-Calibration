# !/usr/bin/env
import rospy
import rospkg
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3, Quaternion, Pose
from mvextractor.videocap import VideoCap

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

rospack = rospkg.RosPack()
path = rospack.get_path('humrs_control')
urdf_file = path + '/../urdf/snake.urdf'

blank_img = 255 * np.ones((480, 640, 3), np.uint8)  # Adjust dimensions as needed
read_img = 255 * np.ones((480, 240, 3), np.uint8)  # Adjust dimensions as needed

def camera_callback(msg):
  global bgr_copy
  global img_step
  global img_stamp

  bgr_copy = np.copy(np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1))
  img_step = msg.step
  img_stamp = msg.header.stamp.to_sec()
  
def getMarkers(img):
  # ArUco marker detection
  arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
  arucoParams = cv2.aruco.DetectorParameters()
  arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
  
  corners, ids, rejects = arucoDetector.detectMarkers(img)
  return corners, ids, rejects

def showMarkers(img, corners, ids, tvecs):

  setMarkerWindow()
  # Loop through each detected marker
  for i in range(len(corners)):
    # Extract current marker info
    corner = corners[i]
    id = ids[i]
    tvec = tvecs[i]

    xTopLeft, yTopLeft = corner[0][0]
    xTopRight, yTopRight = corner[0][1]
    xBottomRight, yBottomRight = corner[0][2]
    xBottomLeft, yBottomLeft = corner[0][3]
    
    # Calculate center coordinates for the current marker
    xCenter = (xTopLeft + xTopRight + xBottomLeft + xBottomRight) / 4
    yCenter = (yTopLeft + yTopRight + yBottomLeft + yBottomRight) / 4

    # Draw the detected marker on the image copy
    cv2.aruco.drawDetectedMarkers(blank_img, [corner], np.array([id]), borderColor=(0, 0, 0))
    cv2.aruco.drawDetectedMarkers(img, [corner],np.array([id]))

    # Calculate distance to the marker based on Z-component of translation vector
    xy_dist = np.sqrt(tvec[0] ** 2 + tvec[1] ** 2)

    # Draw marker information on the image copy
    cv2.circle(blank_img,(int(xCenter),int(yCenter)),5,(0,0,255),-1)
    cv2.line(blank_img,(320,240),(int(xCenter),int(yCenter)),(0,0,0),1)
    cv2.putText(read_img, f"==>Marker ID: {id}", (5 , 10 + i * 30), 1, 0.75, (0, 0, 0))
    cv2.putText(read_img,f"XY Distance: {int(xy_dist)} cm away from tag",(5 , 20 + i * 30), 1, 0.75, (0, 0, 0))
    cv2.putText(read_img,f"Lateral Distance: {int(tvec[2])} cm away from tag",(5 , 30 + i * 30),1,0.75,(0,0,0))
    

  # Display the image with markers and information
  cv2.imshow("Readings", read_img)
  cv2.imshow("Position WRT Camera", blank_img)
  cv2.imshow("Camera Feed", img)

def setMarkerWindow():
    # assumption: camera frame is 480x640
    cv2.rectangle(blank_img,(0,240),(640,480),(0,255,0),-1)
    cv2.rectangle(blank_img,(0,0),(640,240),(255,255,255),-1)
    cv2.circle(blank_img,(320,240),5,(0,0,255),-1)
    cv2.imshow("Position WRT Camera",blank_img)
    
    cv2.rectangle(read_img,(0,0),(240,480),(255,255,255),-1)
    cv2.imshow("Readings", read_img)
    
def main():
  
    rospy.init_node('aruco_pose_estimator', anonymous=True)
    image_sub = rospy.Subscriber('/humrs/camera/image', Image, camera_callback, queue_size=1, buff_size=6220800*2) 
    
    cam = camera(('rtsp://admin:biorobotics!@192.168.1.64:554/Streaming/Channels/101'))
    print('Camera is alive?: %d' %(cam.p.is_alive()))
    rospy.spin()

def image_callback(image_msg):
    
    # Publisher for estimated pose
    pose_publisher = rospy.Publisher("/aruco_pose", Pose, queue_size=1)
    rospy.init_node("position_estimate")
    
    # rate=rospy.Rate(1)
    
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

    # Detect ArUco markers
    corners, ids, rejects = getMarkers(cv_image)
    
    cameraMat = np.array([[1.51319600e3, 0, 1.02309616e+03],
                         [0, 1.51474234e+03, 7.10080186e+02],
                         [0, 0, 1]])
    distCoeffs = np.array([-0.28857924, 0.22533772, 0.00165789, 0.00827434, -0.136742])
    sideLength = 15  # cm (adjust based on your marker size)
    objPoints = np.array([[-sideLength / 2, sideLength / 2,0],
                              [sideLength / 2, sideLength / 2,0],
                              [sideLength / 2, -sideLength / 2,0],
                              [-sideLength / 2, -sideLength / 2,0]], dtype=np.float32)
    while True:
        if len(corners) > 0:  # At least one marker detected
                rvecs = []
                tvecs = []
                for i in range(len(corners)):  # Iterate through each detected marker
                    coord = corners[i]
                    success, rvecs_i, tvecs_i = cv2.solvePnP(objPoints, coord, cameraMat, distCoeffs)

                    if success:
                        rvecs.append(rvecs_i)
                        tvecs.append(tvecs_i)
                        
        # Publish pose as a ROS message
        pose_msg = Pose()
        pose_msg.position.x = tvecs[0][0]
        pose_msg.position.y = tvecs[0][1]
        pose_msg.position.z = tvecs[0][2]
        # Set orientation if needed (using rvecs)
        pose_publisher.publish(pose_msg)

        
        # Visualize markers and pose information (optional for debugging)
        showMarkers(cv_image, corners, ids, tvecs)  # Display all detected markers

        cv2.imshow("Camera Feed", cv_image)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

main()
