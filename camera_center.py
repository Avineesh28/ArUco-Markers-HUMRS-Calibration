import cv2
import time

def main():
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: can't open camera")
        exit()
        
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
        cv2.line(img,(320,0),(320,480),(255,0,0),2)
        cv2.line(img,(0,240),(640,240),(255,0,0),2)
        cv2.imshow("Camera Feed",img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
