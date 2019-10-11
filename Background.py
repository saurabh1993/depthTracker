#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pyrealsense2 as rs

cv2.namedWindow('Background')

def getDepthImage(depth):
    depth_data = depth.as_frame().get_data()
    depthImage = np.asanyarray(depth_data)
    depthImage[depthImage>5000]=0
    depthImage = depthImage/5000*255
    depthImage = depthImage.astype('uint8')
    return depthImage

def callibrateBackGroundmin(num):
    BackgroundImage = np.ones((720,1280))*255
    pipeline = rs.pipeline()
    pipeline.start()
    stay = 1
    nextFrame =1
    
    for i in range(0,num+10):
        
        if stay == 0 or nextFrame ==1:
            frame = pipeline.wait_for_frames()
            depth = frame.get_depth_frame()
            if i>=10:
                nextFrame =0
        #Skip first 10 frames
        if not depth or i<10: continue
        depth_data = depth.as_frame().get_data()
        depthImage = getDepthImage(depth)
        depthImage[depthImage<5]=255
        BackgroundImage =  np.minimum(BackgroundImage,depthImage)
        dispImage = BackgroundImage.copy()
        dispImage[dispImage==255]=0
        cv2.imshow('Background',dispImage)
        key = cv2.waitKey(20)
    
        if key == ord('s'):
            stay = 1 -stay

        if key == ord('f'):
            cv2.destroyAllWindows()
            for i in range (1,30):
                cv2.waitKey(1)
            break

        if key == ord('d'):
            nextFrame =1

    pipeline.stop()
    BackgroundImage[BackgroundImage==255]=0
    return BackgroundImage


BG = callibrateBackGroundmin(50)
cv2.imwrite("BG/bg.jpg",BG)
cv2.destroyAllWindows()
for i in range (0,30):
    cv2.waitKey(1)


# In[ ]:




