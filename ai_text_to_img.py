## import libraries ##
import cv2
import numpy as np
import os 
import Hand_Tracking_Module as htm
import pytesseract
import requests
import io
from PIL import Image
import base64
import api_key

## to list img frm HEADER ##
folderpath = 'HEADER'
lst = os.listdir(folderpath)
#print(lst)
overlaylist = []
for i in lst:
    image = cv2.imread(f'{folderpath}/{i}')
    overlaylist.append(image)
#print(len(overlaylist))    
header = overlaylist[0]

##################################
drawcolor=(255,255,255)
brushthickness=8
eraserthickness=50
##################################

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)

xp, yp = 0, 0 

imgcanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    ### 1-import image ###
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    ### 2-find hand landmarks ###
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)

    if len(lmlist) != 0:
        #print(lmlist)

        ## tip of index and middle fingers ##
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        ### 3-which fingers are up ###
        fingers = detector.fingersUp()
        #print(fingers)    

        ### 4-if selection mode - two fingers are up ###
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            #print("Selection Mode")

            ## checking for the click ##
            if y1 < 125:
                if 210 < x1 < 350:
                    header = overlaylist[1]
                    drawcolor = (255, 255, 255)
                elif 500 < x1 < 630:
                    header = overlaylist[2] 
                    drawcolor = (0, 0, 0) 
            cv2.rectangle (img, (x1, y1 - 25), (x2, y2 + 25), drawcolor, cv2.FILLED)    

        ### 5-if drawing mode - index finger is up ###
        if fingers[1] and fingers[2] == False:
            cv2.circle (img, (x1, y1), 15, drawcolor, cv2.FILLED)
            #print("Drawing Mode") 

            if xp == 0 and yp ==0 :
                xp, yp = x1, y1
            
            if drawcolor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, eraserthickness)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), drawcolor, eraserthickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushthickness)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), drawcolor, brushthickness)

            xp, yp = x1, y1
    
    imggray = cv2.cvtColor(imgcanvas, cv2.COLOR_BGR2GRAY)
    _, imginv = cv2.threshold(imggray, 50, 255, cv2.THRESH_BINARY_INV)
    imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imginv)
    img = cv2.bitwise_or(img, imgcanvas)

    ## setting the header image ##
    img[0:125, 0:1280] = header
    cv2.imshow("IMAGE", img)
    # cv2.imshow("CANVAS", imgcanvas)
    # cv2.imshow("INV", imginv)
    if cv2.waitKey(1 ) & 0XFF == ord('q'):
        break 
cap.release()
cv2.destroyAllWindows()

## to save image ##
cv2.imwrite('D:\DL PROJECT\MEDIAPIPE_PROJECT\OUTPUT\output_demo.jpg', imginv)

## to show the text image ##
img_text = cv2.imread(r'D:\DL PROJECT\MEDIAPIPE_PROJECT\OUTPUT\output_demo.jpg')

## to extract text from image ##
pytesseract.pytesseract.tesseract_cmd = r"c:\Program Files\Tesseract-OCR\tesseract.exe"
text = pytesseract.image_to_string(img_text)
print(text)

## to generate ai image ##
API_URL = "https://api-inference.huggingface.co/models/ZB-Tech/Text-to-Image"

headers = {"Authorization": f"Bearer {api_key.HF_API_KEY}"}

def query(payload):
    response = requests.post(API_URL, headers = headers, json = payload)
    return response.content

image_bytes = query({"inputs": f"3D realistic {text} with cinematic background",})

# You can access the image with PIL.Image for example
image_data = base64.b64decode(image_bytes)
image = Image.open(io.BytesIO(image_bytes))
image.show() 
