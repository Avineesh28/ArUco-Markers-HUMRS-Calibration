import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def getMarkers(img):
    # code referenced from https://stackoverflow.com/questions/74964527/attributeerror-module-cv2-aruco-has-no-attribute-dictionary-get 
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters()
    arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    corners, ids, rejects = arucoDetector.detectMarkers(img)
    return (corners, ids, rejects)

def showMarkers(img,corners, ids, xCenter, yCenter, xyDist, Ztvecs):
    # assumption: camera frame is 480x640
    cv2.rectangle(blank_img,(0,240),(640,480),(0,255,0),-1)
    cv2.rectangle(blank_img,(0,0),(640,240),(255,255,255),-1)
    cv2.circle(blank_img,(320,240),5,(0,0,255),-1)

    cv2.aruco.drawDetectedMarkers(blank_img, corners,ids, borderColor=(0,0,0))
    cv2.aruco.drawDetectedMarkers(img, corners,ids)
    
    cv2.circle(blank_img,(xCenter,yCenter),5,(0,0,255),-1)
    cv2.line(blank_img,(320,240),(xCenter,yCenter),(0,0,0),1)
    
    cv2.putText(blank_img,f"XY Distance: {int(xyDist)} cm away from tag",(5,10),1,0.75,(0,0,0))
    cv2.putText(blank_img,f"Lateral Distance: {int(Ztvecs)} cm away from tag",(5,20),1,0.75,(0,0,0))
    
    cv2.imshow("Position WRT Camera",blank_img)

def setMarkerWindow():
    # assumption: camera frame is 480x640
    cv2.rectangle(blank_img,(0,240),(640,480),(0,255,0),-1)
    cv2.rectangle(blank_img,(0,0),(640,240),(255,255,255),-1)
    cv2.circle(blank_img,(320,240),5,(0,0,255),-1)
    cv2.imshow("Position WRT Camera",blank_img)

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
        cameraMat = np.array([
            [1.51319600e3,0,1.02309616e+03],
            [0,1.51474234e+03, 7.10080186e+02],
            [0,0,1]])
        
        distCoeffs = np.array([-0.28857924,
                                0.22533772, 
                                0.00165789, 
                                0.00827434, 
                                -0.136742])

        sideLength = 10 # cm (units are super unclear so i'm just going with cm)
        objPoints = np.array([[-sideLength / 2, sideLength / 2,0],
                              [sideLength / 2, sideLength / 2,0],
                              [sideLength / 2, -sideLength / 2,0],
                              [-sideLength / 2, -sideLength / 2,0]], dtype=np.float32)
        rvecs = []
        tvecs = []
        
        for coord in corners:
            success, r, t = cv2.solvePnP(objPoints, coord, cameraMat, distCoeffs, False, cv2.SOLVEPNP_IPPE_SQUARE)
            if success!=1:
                print("something went wrong?")
                break
            rvecs.append(r)
            tvecs.append(t)

        if len(rvecs)==1 and len(tvecs)==1:

            # convert rotation vector to rotation matrix
            rmat,jacobian = cv2.Rodrigues(rvecs[0])
            res = np.concatenate((rmat,tvecs[0]),axis=1)
            fullres = np.concatenate((res,np.array([[0,0,0,1]])),axis=0)
            
            # this is the important matrix
            transformMatrix = np.linalg.inv(fullres)

            #get camera frame coord for center of tag
            corner = corners[0][0] #just one tag at a time
            xTopLeft, yTopLeft = corner[0][0], corner[0][1]
            xTopRight,yTopRight = corner[1][0], corner[1][1]
            xBottomRight,yBottomRight = corner[2][0], corner[2][1]
            xBottomLeft,yBottomLeft = corner[3][0], corner[3][1]

            avgX = (xTopLeft+xTopRight+xBottomLeft+xBottomRight)/4
            avgY = (yTopLeft+yTopRight+yBottomLeft+yBottomRight)/4

            xyDist = np.sqrt(tvecs[0][0] ** 2 + tvecs[0][1] ** 2)
            
            xCenter=int((xTopLeft+xTopRight)/2)
            yCenter=int((yBottomLeft+yTopLeft)/2)
           
            # print(tvecs) - Was using to debug the distance calibration
        
            showMarkers(img,corners,ids,xCenter,yCenter,xyDist,tvecs[0][2])
        cv2.imshow("Camera Feed",img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
