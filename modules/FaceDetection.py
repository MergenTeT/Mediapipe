import cv2 as cv
import mediapipe as mp
import time

cTime,pTime =0,0

mpFace = mp.solutions.face_detection
face = mpFace.FaceDetection()
mpDraw = mp.solutions.drawing_utils
path = "humans/1.mp4"
cap = cv.VideoCapture(path)


while cap.isOpened():
    
    ret,frame = cap.read()
    
    if ret:
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face.process(frameRGB)
        
        #print(results.detections)#keyword detection
        
        if results.detections:
            
            
                
            #cx,cy,score=int(lm.x*w),int(lm.y*h),int(lm.score*100)
            for id,detection in enumerate(results.detections):
                h,w,c = frame.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)
                x2,y2 = bbox[2]-bbox[0],bbox[1]-bbox[3]
                cv.rectangle(frame, (bbox), (0,255,0),3)
                #mpDraw.draw_detection(frame, detection)
                #points = detection.location_data.relative_keypoints
                for points in detection.location_data.relative_keypoints:
                    cx,cy = int(points.x*w),int(points.y*h)
                    cv.circle(frame, (cx,cy), 5, (0,0,255))
                    #print(points)
                    #print(id,detection)
                    print(f'{detection.label_id}  {detection.score}')
                    #print(points[1].y)
                    #print(points[2].x)
        
        #FPS +
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(frame,"FPS : "+ str(int(fps)), (10,50), cv.FONT_HERSHEY_COMPLEX_SMALL,color =(255,0,255),fontScale = 2)
        #FPS -
        
        cv.imshow("Video",frame)
        
        if cv.waitKey(27) & 0xFF == ord("q"):
            break

cap.release()
cv.destroyAllWindows()






