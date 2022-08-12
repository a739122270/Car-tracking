#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
# from mainForm import Ui_MainWindow
import cv2 
import numpy as np
import time

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("color: rgb(0, 0, 0);\n"
"background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Feature1 = QtWidgets.QPushButton(self.centralwidget)
        self.Feature1.setGeometry(QtCore.QRect(300, 270, 191, 51))
        self.Feature1.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(0, 0, 127);\n"
"font: 75 11pt \"Arial\";")
        self.Feature1.setObjectName("Feature1")
        self.Feature3_2 = QtWidgets.QPushButton(self.centralwidget)
        self.Feature3_2.setGeometry(QtCore.QRect(300, 360, 191, 51))
        self.Feature3_2.setStyleSheet("background-color: rgb(0, 0, 127);\n"
"color: rgb(255, 255, 255);\n"
"font: 75 11pt \"Arial\";")
        self.Feature3_2.setObjectName("Feature3_2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(210, 60, 411, 71))
        self.label_2.setStyleSheet("font: 18pt \"Arial\";")
        self.label_2.setObjectName("label_2")
        self.Feature3 = QtWidgets.QPushButton(self.centralwidget)
        self.Feature3.setGeometry(QtCore.QRect(300, 460, 191, 51))
        self.Feature3.setStyleSheet("background-color: rgb(0, 0, 127);\n"
"color: rgb(255, 255, 255);\n"
"font: 75 11pt \"Arial\";")
        self.Feature3.setObjectName("Feature3")
        self.Basic = QtWidgets.QPushButton(self.centralwidget)
        self.Basic.setGeometry(QtCore.QRect(300, 180, 191, 51))
        self.Basic.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(0, 0, 127);\n"
"font: 75 11pt \"Arial\";")
        self.Basic.setObjectName("Basic")
        self.Feature3_2.raise_()
        self.Feature1.raise_()
        self.label_2.raise_()
        self.Feature3.raise_()
        self.Basic.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.Feature1.clicked.connect(MainWindow.Feature1_Click)
        self.Feature3_2.clicked.connect(MainWindow.Feature2_Click)
        self.Basic.clicked.connect(MainWindow.Basic_Click)
        self.Feature3.clicked.connect(MainWindow.Feature3_Click)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Feature1.setText(_translate("MainWindow", "Mode2"))
        self.Feature3_2.setText(_translate("MainWindow", "Mode3"))
        self.label_2.setText(_translate("MainWindow", "Computr vision - Coursework"))
        self.Feature3.setText(_translate("MainWindow", "Mode4"))
        self.Basic.setText(_translate("MainWindow", "Mode1"))



def Mode1():
    #Video 
    cap = cv2.VideoCapture("Cv.mp4")
    #Shi-Tomasi corner detection param
    feature_params = dict(maxCorners = 1000, qualityLevel = 0.3, minDistance = 50, blockSize = 5)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    #Pramaeters for tracking
    trackLen = 80
    detectInterval = 5
    tracks = []
    frameIndex = 0
    
    #read first frame of the video
    _, frameStart = cap.read() 
    
    #converting to grayscale
    previousGray = cv2.cvtColor(frameStart,cv2.COLOR_BGR2GRAY)
    
    #Find the features/corners using the Shi-Tomasi method to run on Lucas-Kanade algorithm
    previous = cv2.goodFeaturesToTrack(previousGray, mask = None, **feature_params)
    
    mask = np.zeros_like(frameStart)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(ret == False):
            break
        
        #Converting to grayscale
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if(len(tracks) > 0):
            #points/features from our tracks list, -1 obtains the last element of the list
            initialFeatures = np.float32([tr[-1] for tr in tracks]).reshape(-1,1,2)
            #the next position of the features
            flowForward, status1, err1 = cv2.calcOpticalFlowPyrLK(previousGray, frameGray, initialFeatures, None, **lk_params)
            #the calculated old position of the features based on the newly calculated positions to assure that
            #there is no large leaps or random displacements occuring
            flowBackward, status2, err2 = cv2.calcOpticalFlowPyrLK(frameGray, previousGray, flowForward, None, **lk_params)

            #Compare the initial features to the calculated initial positions
            displacement = abs(initialFeatures-flowBackward).reshape(-1,2).max(-1)
            #If the aboslute total replacement is less than 1 it is considered accurate.
            displacementCheck = displacement < 1

            #A new tracks list
            newTracks = []
        
            #For loop over the tracks list, (x,y) for each feature in the new positions, and our displacement check
            for tr, (x,y), goodCheck in zip(tracks, flowForward.reshape(-1,2), displacementCheck):

                #If the displacement check is bad we have errors, we skip this iteration and go to the next one
                if(not goodCheck):
                    continue
                
                #We append this position to our tracks list
                tr.append((x,y))

                #If how long a features being tracked for is longer than the allowable track length we delete the first position
                #in the track allowing more
                if(len(tr) > trackLen):
                    del tr[0]

                #We append our tracks
                newTracks.append(tr)

                #Draw a circle of thickness 2 with color green at position x,y
                cv2.circle(frame, (int(x),int(y)), 2, (0, 255, 0), -1)
            
            #Assign our new tracks to our tracks list
            tracks = newTracks

            #For every position for each track we draw a line between them.
            cv2.polylines(frame, [np.int32(tr) for tr in tracks], False, (0,255,0))
        
        #This checks what the index is compared to how often we want to resample our features, this also occurs the first frame
        if(frameIndex % detectInterval == 0):

            #Create a mask of pixel values being 255
            mask = np.zeros_like(frameGray)
            mask[:] = 255

            #for every position in track, draw a circle there on our mask, this allows us to resample based on previous features
            for x,y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x,y), 5,0,-1)

            #Using the marked positions on our mask, we run corner detection for updated features, often the same as previous
            #If there is no features, the mask is blank so we calculate features from scratch
            features = cv2.goodFeaturesToTrack(frameGray, mask=mask, **feature_params)

            #If the features are calculated, we append these to our tracks array to use in the above if loop
            if(features is not None):
                for x, y in np.float32(features).reshape(-1,2):
                    tracks.append([(x,y)])

        #Increment index and update the previous frame to be the current frame
        frameIndex = frameIndex +1
        previousGray = frameGray

        #Show the results
        cv2.imshow('Mode1', frame)

        #If q is pressed it breaks out of the process, else it waits 10 milliseconds to proceed to the next image
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
def Mode2():
    cap = cv2.VideoCapture('Cv.mp4')
#     fgbg = cv.createBackgroundSubtractorMOG()
    fgbg = cv2.createBackgroundSubtractorMOG2()
#     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
#     fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
    MaxTracking=4
    while(1):
        ret, frame = cap.read()
        if(ret == False):
            break
        fgmask = fgbg.apply(frame)

        # Max area
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxArea = []
        maxIndex = []
        for l in range(MaxTracking):
            maxArea.append(0)
            maxIndex.append(0)
        
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            for j in range(0,MaxTracking-1):
#                 if area ==maxArea[j]:
#                     break
                if area > maxArea[j]:
                    n=MaxTracking-1
                    k=j
                    while(MaxTracking-k-1>0):
                        maxArea[n] = maxArea[n-1]
                        maxIndex[n] = maxIndex[n-1]
                        n=n-1
                        k=k+1
                    maxArea[n] = area
                    maxIndex[n] =i
                    break                
                
        # draw
#         cv.drawContours(frame, contours, maxIndex, (0, 0, 255), 1)
        # rectangle
        qwe=0
        while qwe<MaxTracking:
            x, y, w, h = cv2.boundingRect(contours[maxIndex[qwe]])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            qwe=qwe+1
#             center_x = int(x + w/2)
#             center_y = int(y + h/2)
#             cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
        
        cv2.imshow('Mode2',fgmask)
        cv2.imshow('Mode2',frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
def Mode3():
    global coor_x,coor_y,coor 
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    camera = cv2.VideoCapture('Cv.mp4') # read video
    fps = camera.get(cv2.CAP_PROP_FPS)# fps

    def OnMouseAction(event,x,y,flags,param):
        global coor_x,coor_y,coor 
        if event == cv2.EVENT_LBUTTONDOWN:
            print("%s" %x,y)
            coor_x ,coor_y = x ,y
            coor_m = [coor_x,coor_y]
            coor = np.row_stack((coor,coor_m))
        elif event==cv2.EVENT_LBUTTONUP:
#             return (coor_x, coor_y)
            cv2.line(img, (coor_x, coor_y), (coor_x, coor_y), (255, 255, 0), 7)
#         elif event==cv2.EVENT_RBUTTONDOWN :
#         elif flags==cv2.EVENT_FLAG_LBUTTON:
#         elif event==cv2.EVENT_MBUTTONDOWN :
    # size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    grabbed, img = camera.read() 
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image',OnMouseAction)
    while(1):
        cv2.imshow('Image',img)
        k=cv2.waitKey(1) & 0xFF
        if k==ord(' '): 
            break
    cv2.destroyAllWindows() 


    feature_params = dict(maxCorners = 1000, qualityLevel = 0.3, minDistance = 50, blockSize = 5)
    lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    # track
    trackLen = 80
    detectInterval = 5
    tracks = []
    frameIndex = 0
    frameStart = img 
    previousGray = cv2.cvtColor(frameStart,cv2.COLOR_BGR2GRAY)
    previous = cv2.goodFeaturesToTrack(previousGray, mask = None, **feature_params)
    mask=np.zeros_like(img[coor[1,1]:coor[2,1], coor[1,0]:coor[2,0]])
    mask1 = np.zeros_like(img[coor[1,1]:coor[2,1], coor[1,0]:coor[2,0]])
    # track end


    Width_choose = coor[2,0]-coor[1,0] 
    Height_choose = coor[2, 1] - coor[1, 1] 
    Video_choose = np.zeros((Width_choose, Height_choose, 3), np.uint8)
    while True:
        grabbed, frame = camera.read() 

        if not grabbed:
            break


        #Converting to grayscale and resizing
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_data = np.array(frameGray)  
        box_data = frame_data[coor[1,1]:coor[2,1], coor[1,0]:coor[2,0]] 
        x = range(Height_choose)
        lwpCV_box = cv2.rectangle(frame, (coor[1,0],coor[1,1]), (coor[2,0],coor[2,1]), (0, 255, 0), 2)

        #Check if we have features
        if(len(tracks) > 0):

            #points/features from our tracks list, -1 obtains the last element of the list
            initialFeatures = np.float32([tr[-1] for tr in tracks]).reshape(-1,1,2)
            #the next position of the features
            flowForward, status1, err1 = cv2.calcOpticalFlowPyrLK(previousGray[coor[1,1]:coor[2,1], coor[1,0]:coor[2,0]], frameGray[coor[1,1]:coor[2,1], coor[1,0]:coor[2,0]], initialFeatures, None, **lk_params)
            #the calculated old position of the features based on the newly calculated positions to assure that
            #there is no large leaps or random displacements occuring
            flowBackward, status2, err2 = cv2.calcOpticalFlowPyrLK(frameGray[coor[1,1]:coor[2,1], coor[1,0]:coor[2,0]], previousGray[coor[1,1]:coor[2,1], coor[1,0]:coor[2,0]], flowForward, None, **lk_params)

            #Compare the initial features to the calculated initial positions
            displacement = abs(initialFeatures-flowBackward).reshape(-1,2).max(-1)
            #If the aboslute total replacement is less than 1 it is considered accurate.
            displacementCheck = displacement < 1

            #A new tracks list
            newTracks = []

            #For loop over the tracks list, (x,y) for each feature in the new positions, and our displacement check
            for tr, (x,y), goodCheck in zip(tracks, flowForward.reshape(-1,2), displacementCheck):

                #If the displacement check is bad we have errors, we skip this iteration and go to the next one
                if(not goodCheck):
                    continue
                #We append this position to our tracks list
                tr.append((x,y))
                #If how long a features being tracked for is longer than the allowable track length we delete the first position
                #in the track allowing more
                if(len(tr) > trackLen):
                    del tr[0]

                #We append our tracks
                newTracks.append(tr)

                #Draw a circle of thickness 2 with color green at position x,y
                cv2.circle(frame, (int(x)+coor[1,0],int(y)+coor[1,1]), 2, (0, 255, 0), -1)

            #Assign our new tracks to our tracks list
            tracks = newTracks

            #For every position for each track we draw a line between them.


            #For every position for each track we draw a line between them.
            trnews=[]
            for tr in tracks:
                for splittr in tr:
    #                 print("111",splittr)
                    splittrlist=list(splittr)
                    splittrlist[0]=splittr[0]+coor[1,0]
                    splittrlist[1]=splittr[1]+coor[1,1]
                    trnews.append(tuple(splittrlist))
    #                 print("222",tuple(splittrlist))
                cv2.polylines(frame, [np.int32(trnews)], False, (0,255,0)) 
                trnews=[]
    #         cv2.polylines(frame, [np.int32(tr) for tr in tracks], False, (0,255,0))
    #         print("222",trnews)


        #This checks what the index is compared to how often we want to resample our features, this also occurs the first frame
        if(frameIndex % detectInterval == 0):

            #Create a mask of pixel values being 255
            mask=np.zeros_like(box_data)
            mask1 = np.zeros_like(frameGray[coor[1,1]:coor[2,1], coor[1,0]:coor[2,0]])
            mask[:] = 255

            #for every position in track, draw a circle there on our mask, this allows us to resample based on previous features
            for x,y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x,y), 5,0,-1)

            #Using the marked positions on our mask, we run corner detection for updated features, often the same as previous
            #If there is no features, the mask is blank so we calculate features from scratch
            features = cv2.goodFeaturesToTrack(frameGray[coor[1,1]:coor[2,1], coor[1,0]:coor[2,0]], mask=mask, **feature_params)

            #If the features are calculated, we append these to our tracks array to use in the above if loop
            if(features is not None):
                for x, y in np.float32(features).reshape(-1,2):
                    tracks.append([(x,y)])

        #Increment index and update the previous frame to be the current frame
        frameIndex = frameIndex +1
        previousGray = frameGray


    #     cv2.imshow('frame',frame)

        cv2.imshow('Mode3', frame) 
    #     cv2.imshow('sum', emptyImage)  
        if cv2.waitKey(30) & 0xFF == ord('q'):
                break
#     out.release()
    camera.release()
    cv2.destroyAllWindows()
    
def Mode4():
    global coor_x,coor_y,coor
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    camera = cv2.VideoCapture('Cv.mp4') 
    fps = camera.get(cv2.CAP_PROP_FPS)


    def OnMouseAction(event,x,y,flags,param):
        global coor_x,coor_y,coor
        if event == cv2.EVENT_LBUTTONDOWN:
            print("%s" %x,y)
            coor_x ,coor_y = x ,y
            coor_m = [coor_x,coor_y]
            coor = np.row_stack((coor,coor_m))
        elif event==cv2.EVENT_LBUTTONUP:
            cv2.line(img, (coor_x, coor_y), (coor_x, coor_y), (255, 255, 0), 7)
#         elif event==cv2.EVENT_RBUTTONDOWN :
#         elif flags==cv2.EVENT_FLAG_LBUTTON:
#         elif event==cv2.EVENT_MBUTTONDOWN :

    grabbed, img = camera.read() 
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image',OnMouseAction)
    while(1):
        cv2.imshow('Image',img)
        k=cv2.waitKey(1) & 0xFF
        if k==ord(' '): 
            break
    cv2.destroyAllWindows() 

    Width_choose = coor[2,0]-coor[1,0] 
    Height_choose = coor[2, 1] - coor[1, 1] 
    Video_choose = np.zeros((Width_choose, Height_choose, 3), np.uint8)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    MaxTracking=2
    while True:
        grabbed, frame = camera.read() 
        if not grabbed:
            break
        gray_lwpCV = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        frame_data = np.array(gray_lwpCV)  
        box_data = frame_data[coor[1,1]:coor[2,1], coor[1,0]:coor[2,0]] 
        x = range(Height_choose)
    #     emptyImage = np.zeros((Width_choose * 10, Height_choose * 2, 3), np.uint8)
    #     Video_choose = frame[coor[1,1]:coor[2,1],coor[1,0]:coor[2,0]]
    #     out.write(Video_choose)
    #     cv2.imshow('Video_choose', Video_choose)
    #     for i in x:
    #         cv2.rectangle(emptyImage, (i*2, (Width_choose-pixel_sum[i]//255)*10), ((i+1)*2, Width_choose*10), (255, 0, 0), 1)
    #     emptyImage = cv2.resize(emptyImage, (320, 240))
        lwpCV_box = cv2.rectangle(frame, (coor[1,0],coor[1,1]), (coor[2,0],coor[2,1]), (0, 255, 0), 2)



        fgmask = fgbg.apply(box_data)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxArea = []
        maxIndex = []
        for l in range(MaxTracking):
            maxArea.append(0)
            maxIndex.append(0)

        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            for j in range(0,MaxTracking-1):
    #                if area ==maxArea[j]:
    #                     break
                if area > maxArea[j]:
                    n=MaxTracking-1
                    k=j
                    while(MaxTracking-k-1>0):
                        maxArea[n] = maxArea[n-1]
                        maxIndex[n] = maxIndex[n-1]
                        n=n-1
                        k=k+1
                    maxArea[n] = area
                    maxIndex[n] =i
                    break                

    #         cv.drawContours(frame, contours, maxIndex, (0, 0, 255), 1)
        qwe=0
        while qwe<MaxTracking:
            x, y, w, h = cv2.boundingRect(contours[maxIndex[qwe]])
            cv2.rectangle(frame, (coor[1,0]+x, coor[1,1]+y), (coor[1,0]+x+w, coor[1,1]+y+h), (0, 255, 0), 1)
            qwe=qwe+1
    #             center_x = int(x + w/2)
    #             center_y = int(y + h/2)
    #             cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)


#         cv2.imshow('Mode4',fgmask)
    #     cv2.imshow('frame',frame)

        cv2.imshow('Mode4', frame) 
        if cv2.waitKey(30) & 0xFF == ord('q'):
                break
#     out.release()
    camera.release()
    cv2.destroyAllWindows()
    
class PyQtMainEntry(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
 
    def Basic_Click(self):
        Mode1()
        
    def Feature1_Click(self):
        Mode2()
        
    def Feature2_Click(self):
        global coor_x,coor_y,coor
        Mode3()
    def Feature3_Click(self):
        Mode4()
        

if __name__ == "__main__":
    coor_x,coor_y = -1, -1 
    coor = np.array([[1,1]]) 
    app = QtWidgets.QApplication(sys.argv)
    window = PyQtMainEntry()
    window.show()
    sys.exit(app.exec_())

