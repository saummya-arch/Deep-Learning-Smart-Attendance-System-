# importing libraries

import cv2
import numpy as np
import face_recognition as face_recog


# img declaration

my = face_recog.load_image_file('sample_images\my.jpg')
my = cv2.cvtColor(my, cv2.COLOR_BGR2RGB)
my_test = face_recog.load_image_file('sample_images\my_test.jpg')
my_test = cv2.cvtColor(my_test, cv2.COLOR_BGR2RGB)


# to show images 
cv2.imshow('main_img', my)
cv2.imshow('test_img', my_test)
cv2.waitKey(1)
cv2.destroyAllWindows()


# finding facelocation 
faceLocation_my = face_recog.face_locations(my)[0] #--> [0] -> mean it is an image
encode_my = face_recog.face_encodings(my)[0]
cv2.rectangle(my, (faceLocation_my[3], faceLocation_my[0]), (faceLocation_my[1], faceLocation_my[2]), (255, 0, 255), 3)

faceLocation_mytest = face_recog.face_locations(my_test)[0] #--> [0] -> mean it is an image
encode_mytest = face_recog.face_encodings(my_test)[0]
cv2.rectangle(my_test, (faceLocation_my[3], faceLocation_my[0]), (faceLocation_my[1], faceLocation_my[2]), (255, 0, 255), 3)

#print(encode_my)
#print(encode_mytest)

# prinitng result if face matches
results = face_recog.compare_faces([encode_my], encode_mytest)
print(results)