import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

framedWidth = 1920
frameHeight = 1080
cap.set(3,framedWidth)
cap.set(4,frameHeight)
cap.set(10,150)

while(True):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if(results.multi_face_landmarks):
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS {int(fps)}', (10,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(255,0,0))
    cv2.imshow("Image", img)
    if(cv2.waitKey(1)==27):
        break

cap.release()