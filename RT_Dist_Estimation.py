import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

'''
-----Pose Estimation usign ArUco Tags-----
Inputs
    image "img" of type Image
    camera matrix "cameraMat" of type np.ndarray
    distortion coefficient matrix "distCoeffs" of type np.ndarray
    real-world tag side length "sideLength" of type int32

Outputs (printed for now)
    coordinates "worldCoords" of type np.ndarray

'''

blank_img = 255 * np.ones((480, 640, 3), np.uint8)  # Adjust dimensions as needed
read_img = 255 * np.ones((480, 240, 3), np.uint8)  # Adjust dimensions as needed

def getMarkers(img):
    # code referenced from https://stackoverflow.com/questions/74964527/attributeerror-module-cv2-aruco-has-no-attribute-dictionary-get 
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters()
    arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    corners, ids, rejects = arucoDetector.detectMarkers(img)
    return (corners, ids, rejects)

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
    cv2.putText(read_img, f"\n==>Marker ID: {id}", (5 , 10 + i * 30), 1, 0.75, (0, 0, 0))
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
    
    cap = cv2.VideoCapture(0)
    setMarkerWindow()
    if not cap.isOpened():
        print("Error: can't open camera")
        exit()
        
    while True:
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Crop to Robot Specs -- Will need to remove for actual implementation
        # img = img[89:391, 53:587]   

        corners, ids, rejects = getMarkers(img)
        
        # using camera matrix and distance coefficients from humrs_vrep/humrs_control/config/external_camera_underwater.yaml (pipe_entry branch)
        
        #Calibration for the camera on the robot
        # cameraMat = np.array([
        #     [1.51319600e3,0,1.02309616e+03],
        #     [0,1.51474234e+03, 7.10080186e+02],
        #     [0,0,1]])

        cameraMat = np.array([
            [592.30698448, 0, 314.51159249],
            [0,591.37584641, 234.58324437],
            [0,0,1]])
        
        distCoeffs = np.array([-0.28857924,
                                0.22533772, 
                                0.00165789, 
                                0.00827434, 
                                -0.136742])

        sideLength = 15 # cm ~ Need to adjust with every testing stage for accurate distance calculation
        objPoints = np.array([[-sideLength / 2, sideLength / 2,0],
                              [sideLength / 2, sideLength / 2,0],
                              [sideLength / 2, -sideLength / 2,0],
                              [-sideLength / 2, -sideLength / 2,0]], dtype=np.float32)
        if len(corners) > 0:  # At least one marker detected
            rvecs = []
            tvecs = []
            for i in range(len(corners)):  # Iterate through each detected marker
                coord = corners[i]
                success, rvecs_i, tvecs_i = cv2.solvePnP(objPoints, coord, cameraMat, distCoeffs)

                if success:
                    rvecs.append(rvecs_i)
                    tvecs.append(tvecs_i)

            # if len(rvecs)==1 and len(tvecs)==1:

            #     # convert rotation vector to rotation matrix
            #     rmat,jacobian = cv2.Rodrigues(rvecs[0])
            #     res = np.concatenate((rmat,tvecs[0]),axis=1)
            #     fullres = np.concatenate((res,np.array([[0,0,0,1]])),axis=0)
                
            #     # this is the important matrix
            #     transformMatrix = np.linalg.inv(fullres)
        
            showMarkers(img, corners, ids, tvecs)  # Display all detected markers

        cv2.imshow("Camera Feed",img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
