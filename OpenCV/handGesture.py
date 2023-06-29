import mediapipe as mp
import cv2
import keyboard

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

id_list = [4, 8]
point1_coord = []
point2_coord = []

trigger_msg = "Thumb and Index hit"
button_coordinates = (100,100,150,150)
buttons = []

class Button:
    def __init__(self,text):
        self.text = text

    def changeName(self):
        self.text = "B2"
    
    def get_name(self):
        print(self.text)

    def display(self, img):
        #Button
        button_color = (255,0,0)
        button_thickness = -1
        #Text
        text = self.text
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (0, 0, 0)
        font_thickness = 2
        cv2.rectangle(img,(button_coordinates[0], button_coordinates[1]),(button_coordinates[2], button_coordinates[3]), button_color, button_thickness)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = int((button_coordinates[2] + button_coordinates[0] - text_size[0][0]) / 2)
        text_y = int((button_coordinates[3] + button_coordinates[1] + text_size[0][1]) / 2)
        cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
    
while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                if(id in id_list):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if id == 4:
                        point1_coord = [cx, cy]
                    elif id == 8:
                        point2_coord = [cx, cy]
                    
                    if point1_coord and point2_coord:
                        if ((abs(point1_coord[0] - point2_coord[0]) < 30 ) and (abs(point1_coord[1] - point2_coord[1]) < 30)):
                            cv2.putText(img, trigger_msg, (350, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 3)
                            button = Button("B1")
                            if(len(buttons) == 0):
                                buttons.append(button)
                        if(button_coordinates[0]<point1_coord[0]<button_coordinates[2] and button_coordinates[1]<point1_coord[1]<button_coordinates[2]):
                            print("finger hit the button")
                            button.get_name()
                            button.changeName()
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    if(buttons):
        buttons[0].display(img)
    cv2.putText(img, str(int(cap.get(cv2.CAP_PROP_FPS))), (20, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    if(keyboard.is_pressed('d')):
        if(len(buttons) == 1):
            del buttons[0]
        else:
            print("There's nothing to delete !")
            
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
