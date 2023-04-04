import cv2 as cv
import mediapipe as mp
import time

path = "humans/2.mp4"
cap = cv.VideoCapture(path)
cTime,pTime=0,0
mpMesh = mp.solutions.face_mesh
faceMesh = mpMesh.FaceMesh()
mpDraw = mp.solutions.drawing_utils
while cap.isOpened():
    
    ret,frame = cap.read()
    if ret:
        
        results = faceMesh.process(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        h,w,c = frame.shape
        if results.multi_face_landmarks:
            for id,lm in enumerate(results.multi_face_landmarks[0].landmark):
                cx,cy = int(lm.x*w),int(lm.y*h)
                cv.circle(frame, (cx,cy), 5, (0,0,255))
                print(id)
                print(lm)
                
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(frame,"FPS : " +str(int(fps)), (10,50), cv.FONT_HERSHEY_PLAIN, 3, (0,255,0))
        cv.imshow("Frame", frame)
        
        if cv.waitKey(27) & 0xFF == ord("q"):
            break

cap.release()
cv.destroyAllWindows()






