import cv2 as cv
import mediapipe as mp
import time



class poseDetector():
    def __init__(self,mode = False,model_complexity=1,
                         smooth_landmarks=True,
                         enable_segmentation=False,
                         smooth_segmentation=True,
                         min_detection_confidence=0.5,
                         min_tracking_confidence=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.model_complexity,
                                     self.smooth_landmarks,self.enable_segmentation,
                                     self.smooth_segmentation ,self.min_detection_confidence,
                                     self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        
        
    def findPose(self,frame,draw = True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(frameRGB)
            
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return self.results,frame
        
    def findPosition(self,frame):
        lmList = []
            
        if self.results.pose_landmarks:
            myPose = self.results.pose_landmarks.landmark
            for id,lm in enumerate(myPose):
                visibility = lm.visibility
                h,w,c = frame.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                #if visibility>0.80:
                        #self.mpDraw.draw_landmarks(frame, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
                lmList.append([id,cx,cy,visibility])
            return lmList









def main():
    pTime = 0
    cTime = 0
    path =  r"PoseVideos\3.mp4"
    cap = cv.VideoCapture(0)
    detector = poseDetector()
    
    while cap.isOpened():
        
        ret,frame = cap.read()
        
        
        
       
        
        if ret:
            
            results,frame = detector.findPose(frame)
            lmList = detector.findPosition(frame) 
            if results.pose_landmarks:    
                print(lmList[1])
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            
            cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255),3)
            
            cv.imshow("Video",frame)
            
        if cv.waitKey(27) & 0xFF == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
            
if __name__ == "__main__":
    
    main()