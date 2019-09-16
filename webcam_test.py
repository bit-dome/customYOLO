import sys, os
import glob
import time
import cv2

from yolov3_core import YOLO



def test(yolo):


    # ─── WEBCAM STUFF ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    ret, frame = cap.read()


    # ─── MAIN LOOP ──────────────────────────────────────────────────────────────────
    while ret:       
        ret, frame = cap.read()
        
        t1 = time.time()
               
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        out_image = yolo.detect_image(frame,filename=None)
        out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)       
                
        
        fps = 1 // (time.time() - t1)
        cv2.putText(out_image,"FPS: {:.2f}".format(fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),3)       
        
        
        cv2.imshow("",out_image)
        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            cap.release()
       
    yolo.close_session()






FLAGS = None
if __name__ == '__main__':
    
    test(YOLO())    
 