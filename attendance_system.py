# importing libraries

from ast import While
import cv2
from cv2 import line
import numpy as np
import face_recognition as face_recog
import os
import datetime
import pyttsx3 as textSpeech

engine = textSpeech.init() 

path = 'student_images'

studentImg = []
studentName = []
myList= os.listdir(path)
print(myList) 

for i in myList:
    curImg = cv2.imread(f'{path}/{i}')
    studentImg.append(curImg)
    studentName.append(os.path.splitext(i)[0])

#print(studentName) 


# making function for reading the image and encoding it
def findEncoding(images):
    encoding_list = []
    for i in images:
        img = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encode_img = face_recog.face_encodings(i)[0]
        encoding_list.append(encode_img)
    return encoding_list

# encode list of all the images
EncodeList = findEncoding(studentImg)



# functon for putting names in csv file
def MarkAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()    # to read all lines in csv file
        nameList = []
        for i in myDataList:
            entry = i.split(',')
            nameList.append(entry[0])


        if name not in nameList:      # if name is not in csv file just put it.
            now = datetime.datetime.now()
            f.writelines(f'\n{name}, {now}')
            engine.say('Welcome to class' + name)
            engine.runAndWait()




vid = cv2.VideoCapture(0) # (0) -> shows the source video is from inbuilt camera in system



while True:
    success, frame = vid.read()
    Smaller_frames = cv2.resize(frame, (0,0), None, 0.25, 0.25)

    facesInFrame = face_recog.face_locations(Smaller_frames)
    encodeFacesInFrame = face_recog.face_encodings(Smaller_frames, facesInFrame)

    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame):
        matches = face_recog.compare_faces(EncodeList, encodeFace)
        facedist = face_recog.face_distance(EncodeList, encodeFace)
        
        print(facedist)
        matchIndex = np.argmin(facedist) # -> min dst. atrix is choosen for better accuracy



        if matches[matchIndex]:
            name = studentName[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame,  (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            MarkAttendance(name)
    
    cv2.imshow('Video', frame)
    cv2.waitKey(1)