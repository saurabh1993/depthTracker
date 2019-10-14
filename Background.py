#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import cv2
import pyrealsense2 as rs

cv2.namedWindow('Background')

def getDepthImage(depth):
    depth_data = depth.as_frame().get_data()
    depthImage = np.asanyarray(depth_data)
    depthImage[depthImage>5000]=0
    depthImage = 1.0*depthImage/5000*255
    depthImage = depthImage.astype('uint8')
    return depthImage

def callibrateBackGroundmin(num):
    BackgroundImage = np.ones((720,1280))*255.0
    pipeline = rs.pipeline()
    pipeline.start()
    stay = 1
    nextFrame =1
    num_max =50
    i=0
    while(True):
        
        if stay == 0 or nextFrame ==1:
            frame = pipeline.wait_for_frames()
            depth = frame.get_depth_frame()
            i +=1
            if i>=10:
                nextFrame =0
        
        #Skip first 10 frames
        if not depth or i<10: continue
        depthImage = getDepthImage(depth)
        depthImage[depthImage<5]=255
        BackgroundImage =  np.minimum(BackgroundImage,depthImage)
        dispImage = BackgroundImage.copy()
        dispImage[dispImage==255]=0
        cv2.imshow('Background',dispImage.astype('uint8'))
        key = cv2.waitKey(20)
        #Start-stop the feed -press 's'
        if key == ord('s'):
            stay = 1 -stay
        
        #Complete callibration -Press 'f'
        if key == ord('f'):
            
            BackgroundImage[BackgroundImage==255]=0
            pipeline.stop()
            return BackgroundImage
        
        #Jump to next frame -press 'd'
        if key == ord('d'):
            nextFrame =1
        
        # Break =press 'g'    
        if key == ord('g'):
            pipeline.stop()
            break
            
    return None
      


BG = callibrateBackGroundmin(50)


if BG is not None:
    print("updated Background")
    cv2.imwrite("bg.jpg",BG)

cv2.destroyAllWindows()
for i in range (0,30):
    cv2.waitKey(1)





