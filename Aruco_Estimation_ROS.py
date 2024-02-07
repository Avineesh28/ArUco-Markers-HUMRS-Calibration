import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3, Quaternion, Pose


blank_img = 255 * np.ones((480, 640, 3), np.uint8)  # Adjust dimensions as needed
read_img = 255 * np.ones((480, 240, 3), np.uint8)  # Adjust dimensions as needed

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
    rospy.init_node('aruco_pose_estimator')
    
    # Subscriber for camera image
    image_sub = rospy.Subscriber("/camera/image_raw", Image, image_callback)  

    rospy.spin()

def image_callback(image_msg):
    
    # Publisher for estimated pose
    pose_publisher = rospy.Publisher("/aruco_pose", Pose, queue_size=10)
    
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

    # Detect ArUco markers
    corners, ids, rejects = getMarkers(cv_image)
    
    cameraMat = np.array([[1.51319600e3, 0, 1.02309616e+03],
                         [0, 1.51474234e+03, 7.10080186e+02],
                         [0, 0, 1]])
    distCoeffs = np.array([-0.28857924, 0.22533772, 0.00165789, 0.00827434, -0.136742])
    

    sideLength = 10  # cm (adjust based on your marker size)
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
