import cv2 as cv
import mediapipe as mp
import time

class faceDetection():
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        
        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(self.min_detection_confidence,self.model_selection)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findFace(self,frame,draw = True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.face.process(frameRGB)
        
        if self.results.detections:
            for detection in self.results.detections:
                if draw:
                    self.mpDraw.draw_detection(frame,detection)
            
            
    def findPosition(self,frame,draw = True): 
        lmList = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                
                h,w,c = frame.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
                lmList.append([id,bbox,int(detection.score[0]*100)])
                if draw:
                    cv.rectangle(frame, bbox, (0,0,255))
                    #print(f'{detection.label_id}  {detection.score}') 
            for points in detection.location_data.relative_keypoints:
                cx,cy = int(points.x*w),int(points.y*h)
                cv.circle(frame, (cx,cy), 5, (0,0,255))
                    #print(id,detection)
                    #print(points)
                    
                    
            return lmList

def main():
    path = "humans/2.mp4"
    cap = cv.VideoCapture(path)
    cTime,pTime = 0,0
    
    detector = faceDetection()
    
    while cap.isOpened():
        
        ret,frame = cap.read()
        if ret:
            detector.findFace(frame,False)
            
            lmList = detector.findPosition(frame,True)
            
            print(lmList)
            
            
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            
            cv.putText(frame,"FPS : " + str(int(fps)), (10,50),cv.FONT_HERSHEY_PLAIN, 3, (0,255,0))
            
            
            cv.imshow("frame", frame)
            
            if cv.waitKey(27) & 0xFF == ord("q"):
                break
    cap.release()
    cv.destroyAllWindows()



if __name__== "__main__":
    main()