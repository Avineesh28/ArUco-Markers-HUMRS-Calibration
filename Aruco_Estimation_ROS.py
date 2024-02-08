import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Quaternion

from rtclib import openRTSP  # Assuming this replaces videocap for RTSP streaming

# Camera class (optional, consider simpler functions)
class Camera:
    def __init__(self, rtsp_url):
        self.cap = openRTSP(rtsp_url)
        self.ret, self.frame = self.cap.read()

    def get_frame(self):
        while not self.ret:
            self.ret, self.frame = self.cap.read()
        return self.frame

    def release(self):
        self.cap.release()

# Global Constants

cam = Camera(rtsp_url="rtsp://admin:biorobotics!@192.168.1.64:554/Streaming/Channels/101")

# Calibration parameters
camera_mat = np.array([[1.51319600e3,0,1.02309616e+03],
                          [0,1.51474234e+03, 7.10080186e+02],
                          [0,0,1]])
dist_coeffs = np.array([-0.28857924,
                                0.22533772, 
                                0.00165789, 
                                0.00827434, 
                                -0.136742])
marker_size = 15  # Adjust based on your marker size (cm)

def aruco_pose_estimator():
    rospy.init_node("aruco_pose_estimator")

    # Camera setup
    
    if not cam.cap.isOpened():
        rospy.logerr("Failed to open camera stream")
        return

    # Image subscriber
    image_sub = rospy.Subscriber("/humrs/camera/image", Image, image_callback)

def image_callback(msg):
  try:
    while not rospy.is_shutdown():
            ret, frame = cam.get_frame()
            if not ret:
                continue

    # ArUco marker detection
    corners, ids, rejects = get_aruco_markers(frame)


    # Publisher for estimated pose
    pose_publisher = rospy.Publisher("/aruco_pose", PoseStamped, queue_size=1)

    if any(ids):
        # Pose estimation for each detected marker
        for i in range(len(corners)):
            tvec, rvec = estimate_pose(corners[i], ids[i], camera_mat, dist_coeffs, marker_size)
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


def estimate_pose(corner, id, camera_mat, dist_coeffs, sideLength):
    obj_points = np.array([[-sideLength / 2, sideLength / 2,0],
                           [sideLength / 2, sideLength / 2,0],
                           [sideLength / 2, -sideLength / 2,0],
                           [-sideLength / 2, -sideLength / 2,0]], dtype=np.float32)
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

  # Get the quaternion from rvec (using the function you provided earlier)
  pose_msg.pose.orientation.x = quaternion[0]  # Set orientation from quaternion
  pose_msg.pose.orientation.y = quaternion[1]
  pose_msg.pose.orientation.z = quaternion[2]
  pose_msg.pose.orientation.w = quaternion[3]

  # Publish the message
  publisher.publish(pose_msg)

if __name__ == "__main__":
    aruco_pose_estimator()
