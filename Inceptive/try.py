import cv2

from random import randrange


trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

img = cv2.imread('Dev.png')

grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

for (x,y,w,h) in face_coordinates:    
    cv2.rectangle(img, (x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)), 2)

# (x,y,w,h) = face_coordinates [0]
# cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0), 5)

# print(face_coordinates)



cv2.imshow('Clever Anurag Face Detector',img)
cv2.waitKey()







print("Code Completed");
