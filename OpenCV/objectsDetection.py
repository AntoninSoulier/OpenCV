import cv2

#Charger l'image que l'on veut analyser
image = cv2.imread('Side Projects\OpenCV\Data_Images\objects.png')
cv2.imshow("Initial", image)

#Appliquer le flou Gaussien pour réduire le bruit
image = cv2.GaussianBlur(image, (5,5), 0)

#Appliquer le detecteur de contours de Canny
edges = cv2.Canny(image, 100, 200)

#Appliquer une dilatation pour connecter les contours
kernel= cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
edges = cv2.dilate(edges, kernel)

#Trouver les contours dans l'image dilatée
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Dessiner les contours sur l'image d'origine
cv2.drawContours(image, contours, -1, (0,255,0), 2)

#Afficher le résultat avec les contours dessinés
cv2.imshow("Contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()