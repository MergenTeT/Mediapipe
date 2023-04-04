import cv2 as cv
import mediapipe as mp
import time

path = r"PoseVideos\2.mp4"
cap = cv.VideoCapture(path)
pTime,cTime = 0,0




mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

while cap.isOpened():
    ret,frame = cap.read()
    
    if ret:
        
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(frame)# görsel işleniyor(RGB formatta)
        
        if results.pose_landmarks:
            # if results.pose_landmarks[3]>0.7:
            print(results.pose_landmarks)#keyword results.pose_landmarks
               
            for id,lm in enumerate(results.pose_landmarks.landmark):
                visibility = lm.visibility
                h,w,c = frame.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                if visibility>0.80:
                    mpDraw.draw_landmarks(frame, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
                    #print(f'{visibility} {cx} {cy}')
                
        #print(results.pose_landmarks)
        
                
        
        
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        cv.putText(frame, str(int(fps)),(10,70),cv.FONT_HERSHEY_COMPLEX,3,(0,255,0))
        #Icv.putText(img, text, org, fontFace, fontScale, color)
        frame = cv.resize(frame,(1080,720))
        cv.imshow("Video",frame)
        
    if cv.waitKey(27) & 0xFF == ord("q"): break
    
cap.release()
cv.destroyAllWindows()








