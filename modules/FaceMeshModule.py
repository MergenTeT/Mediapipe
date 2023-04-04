import cv2 as cv
import mediapipe as mp
import time
import numpy as np
#


class FaceMeshDetection():
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.static_image_mode=static_image_mode
        self.max_num_faces=max_num_faces
        self.refine_landmarks=refine_landmarks
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence
        
        self.mpFaceMesh =mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode,
                                                 self.max_num_faces,
                                                 self.refine_landmarks,
                                                 self.min_detection_confidence,
                                                 self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        
    
    def findFaceMesh(self,frame,draw = True):
        
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(frameRGB)
        
        if self.results.multi_face_landmarks:
            for detection in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, detection)
        return self.results
        
    def findLocation(self,frame,draw = True):
        lmList = []
        if self.results.multi_face_landmarks:
            for id,lm in enumerate(self.results.multi_face_landmarks[0].landmark):
                
                h,w,c = frame.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                print(id,lm)
                lmList.append([id,cx,cy])
            return lmList
        
    # def drawBlackBackground(self,frame,blank):
        
    #     if self.results.multi_face_landmarks:
    #         for lm in self.results.multi_face_landmarks[0].landmark:
    #             #print(id,lm)
    #             h,w,c = frame.shape
    #             cx,cy = int(lm.x*w),int(lm.y*h)
        
                
                
    #             self.bitwise = cv.bitwise_and(frame,blank)
    #             cv.circle(self.bitwise, (cx,cy), 1, (255,255,255),cv.FILLED)
                
                
    #     return self.bitwise
                
        
        

def main():
    path = "humans/3.mp4"
    cap = cv.VideoCapture(0)
    detector = FaceMeshDetection()
    cTime,pTime=0,0
    
    while cap.isOpened():
        
        ret,frame = cap.read()
        # blank = np.zeros(frame.shape,"uint8")
        if ret:
            results = detector.findFaceMesh(frame)
            lmList = detector.findLocation(frame)
            # bitwise = detector.drawBlackBackground(frame,blank)
            # cv.imshow("Mask",bitwise)
            
            #if results.multi_face_landmarks:
                #print(lmList[1])
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv.putText(frame,"FPS : " +str(int(fps)), (10,50), cv.FONT_HERSHEY_PLAIN, 3, (0,255,0))
            cv.imshow("Frame", frame)
            
            if cv.waitKey(27) & 0xFF == ord("q"):
                break
    
    cap.release()
    cv.destroyAllWindows()
    
    
    

if __name__ == "__main__":
    
    main()




