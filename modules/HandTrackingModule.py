import time
import mediapipe as mp
import cv2 as cv



class handDetector():
    def __init__(self,mode = False,maxHands = 2,modelC=1, detectionCon = 0.5,trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelC,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils# noktaları birlestirmek icin
    
    def findHands(self,frame,draw = True):
        frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)#frame ler rgb formatında işleme alınır mediapipe modulunde
        self.results = self.hands.process(frameRGB)
        
        if self.results.multi_hand_landmarks:# tespit edilirse True doner
            for hadsLms in self.results.multi_hand_landmarks:
               if draw:
                   self.mpDraw.draw_landmarks(frame,hadsLms,self.mpHands.HAND_CONNECTIONS)#nokta ve iskelet kismi burada ciziliyor
        return self.results
    
    def findPosition(self,frame,handNo = 0,draw = True):
        
        lmList = []
        if self.results.multi_hand_landmarks:
            myHands = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHands.landmark):
                h,w,c = frame.shape
                cx,cy = int(lm.x*w),int(lm.y*h)#decimal konum koordinatları donduren hadsLms.landmark , frame konum koordinatlarına ceviriyoruz
                #print(id,cx,cy)#konum kooordinatlari 
                lmList.append([id,cx,cy])
                #if draw:
                    #cv.circle(frame, (cx,cy), 5, (0,0,255),cv.FILLED)# nokta kısmı burada ciziliyor
            return lmList
                              
def main():
    pTime = 0
    cTime = 0
    #path = Null
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while cap.isOpened():
        ret,frame = cap.read()
        results = detector.findHands(frame)#draw parametresi False yaparsak ekrana iskelet cizimi cıkmaz
        lmList = detector.findPosition(frame)
        
        
        if results.multi_hand_landmarks:#Yanlızca secilen noktanın bilgilerini veriyor
            if len(lmList) != 0:
                print(lmList[0])
        
        if ret:
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