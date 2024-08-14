### HAND TRACKING MODULE ###

import mediapipe as mp
import cv2


class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
        self.tipIds = [ 4, 8 , 12, 16, 20]

    def findHands(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print (results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        
        return img
    

    def findPosition(self, img, handNo = 0):
        
        self.lmlist = []
        
        if self.results.multi_hand_landmarks: 
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                #print("id :", id, "||", "landmarks :", lm) 
                    
                h, w, c = img.shape
                cx, cy = int (lm.x*w), int (lm.y*h)
                #print(id, ":", cx, cy)
                self.lmlist.append([id, cx, cy])

        return self.lmlist    
         
    def fingersUp(self):
        
        fingers = []

        ## thumb ##
        if self.lmlist[20] [1] < self.lmlist[ 12] [1]:  ## right hand
            if self.lmlist[ self.tipIds [0]] [1] > self.lmlist[ self. tipIds [0] -1] [1]:
                fingers.append(1)
            else:
                fingers.append(0)   
        else:                                           ## left hand
            if self.lmlist[ self.tipIds [0]] [1] < self.lmlist[ self.tipIds [0] -1] [1]:
                fingers.append(1)
            else:
                fingers.append(0)    

        ## other fingers ## 
        for i in range( 1, 5):
            if self.lmlist[ self.tipIds[i]] [2] < self.lmlist[ self.tipIds[i] -2] [2]:
                fingers.append(1)
            else:
                fingers.append(0)   

        return fingers        


def main():
    ## to capture video from webcam

    cap = cv2.VideoCapture(0)

    detector = handDetector()
    
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        
        if len(lmlist) != 0:
            print (lmlist[4])


        cv2.imshow("IMAGE",img)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break



if __name__ == "__main__":
    main()