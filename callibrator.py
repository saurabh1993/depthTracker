#!/usr/bin/env python
# coding: utf-8

# In[13]:


#import packages
import cv2
import numpy as np
import imutils
import sys
import json
#import pyrealsense as rs

#Kernel thresholds

KERNEL_MORPH_DILATE_SIZE=25
KERNEL_DILATE_ITER=5
CONTOUR_AREA_MIN=70
CONTOUR_AREA_MAX=180
#DEPTH_IMAGE_MAX=5000
DEPTH_MASK_THRESH=150
THRESH_MAX_VALUE = 230
THRESH_RETAIN_PEAKS = 230
KERNEL_MORPH_OPEN_SIZE=9
THRESH_RATIO_AREA=0.55
THRESH_WH_RATIO=0.6
ROI_STARTX=10
ROI_STARTY=10
ROI_ENDX=700
ROI_ENDY=1280
TRACKER_RETAIN_FRAMES=10
TRACKER_MAX_DIST_THESH=80
TRACKER_LINE_BOUNDARY=300

depth = None



def getDepthImage(depth):
    depth_data = depth.as_frame().get_data()
    depthImage = np.asanyarray(depth_data)
    depthImage[depthImage>DEPTH_IMAGE_MAX]=0
    depthImage = 1.0*depthImage/DEPTH_IMAGE_MAX*255
    depthImage = depthImage.astype('uint8')
    return depthImage

def displayx(img,title,show=0):
    if show==0 :
        return
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(title) 
    plt.show()

def grabFrame(pipeline ,depth = True,rgb=True):
    frames = pipeline.wait_for_frames()
    if depth==True:
        depth = frames.get_depth_frame()
        depthImage = getDepthImage(depth)
    if rgb == True:
        color = frames.get_color_frame()
        color_data = color.as_frame().get_data()
        colorImage = np.asanyarray(color_data)
    
    #if not depth or not color : continue
    return depthImage,colorImage 


def findHead(image,thresh_rem=THRESH_RETAIN_PEAKS, show= 0):
    
    kernel_size=KERNEL_MORPH_DILATE_SIZE
    kernel_size_open= KERNEL_MORPH_OPEN_SIZE
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    kernel2 = np.ones((kernel_size_open,kernel_size_open),np.uint8)

    
    #Removes extremely high values from image
    image[image>THRESH_MAX_VALUE]=0
    #displayx(image,"orig",show)
    thresh = cv2.threshold(image, DEPTH_MASK_THRESH, 255,cv2.THRESH_BINARY)[1]
    #displayx(thresh,"mask",show)
    thresh[thresh==255]=1
    
    #Apply dilate op-> replace all the values with local maximaum
    dilated = cv2.dilate(image,kernel,iterations=KERNEL_DILATE_ITER)
    #displayx(dilated,"dilated",show)
    
    #Aplly mask on dilated image
    res= dilated*thresh
    #displayx(res,"final dilated",show)
    
    #Subtract the dilated image from original->the local maxima valuea will be ~= 0
    res= res-image
    #displayx(res,"",show)
    
    #replace the unmasked values with max so that we get boundary of maxima
    res[thresh==0]=255
    #displayx(res,"",show)
   
    #Inverse the image-> maxima become highlighted
    res= 255-res
    #displayx(res,"",show)
    
    #Apply threshold to extract just the maxima(top values)
    res[res< thresh_rem] =0
    #displayx(res,"",show)
    
    #Apply closing operation to remove holes from the max values
    res= cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel2)
    
    #displayx(res,"final",show)
    return res


# create trackbars for adjusting thresholds

def KERNEL_MORPH_DILATE_SIZE_CHANGE(x):
    global KERNEL_MORPH_DILATE_SIZE
    if x%2==0:
        x=x+1
    KERNEL_MORPH_DILATE_SIZE=x
    #print()
    #process(depth)
    
def KERNEL_DILATE_ITER_CHANGE(x):
    global KERNEL_DILATE_ITER
    KERNEL_DILATE_ITER=x
    
    
def CONTOUR_AREA_MIN_CHANGE(x):
    global CONTOUR_AREA_MIN
    CONTOUR_AREA_MIN=x

    
def CONTOUR_AREA_MAX_CHANGE(x):
    global CONTOUR_AREA_MAX
    CONTOUR_AREA_MAX=x

    
def DEPTH_MASK_THRESH_CHANGE(x):
    global DEPTH_MASK_THRESH
    DEPTH_MASK_THRESH=x

    
def THRESH_RETAIN_PEAKS_CHANGE(x):
    global THRESH_RETAIN_PEAKS
    THRESH_RETAIN_PEAKS=x

def KERNEL_MORPH_OPEN_SIZE_CHANGE(x):
    global KERNEL_MORPH_OPEN_SIZE
    if x%2==0:
        x=x+1
    KERNEL_MORPH_OPEN_SIZE=x

def THRESH_MAX_VALUE_CHANGE(x):
    global THRESH_MAX_VALUE
    THRESH_MAX_VALUE = x

def ROI_START_X_CHANGE(x):
    global ROI_STARTX
    ROI_STARTX =x

def ROI_START_Y_CHANGE(x):
    global ROI_STARTY
    ROI_STARTY =x

def ROI_END_X_CHANGE(x):
    global ROI_ENDX
    ROI_ENDX =x

def ROI_END_Y_CHANGE(x):
    global ROI_ENDY
    ROI_ENDY =x
    
def preprocess(cnts):
    circles=[]
    for cnt in cnts:
        #get the area
        area = cv2.contourArea(cnt)
        # apply the threshold to get contours with reasonable area
        if area > CONTOUR_AREA_MIN*CONTOUR_AREA_MIN  and area < CONTOUR_AREA_MAX*CONTOUR_AREA_MAX:
            x,y,w,h = cv2.boundingRect(cnt)
            wh_ratio=1.0*w/h
            #Check width to height ratio is in reasonable range
            if(wh_ratio> THRESH_WH_RATIO and wh_ratio < 1/THRESH_WH_RATIO):
                circle = cv2.minEnclosingCircle(cnt)
                ((cx,cy),r)=circle
                circle_area = 22/7*r*r
                #Check if the actual area and enclosing circle area is in reasonable range
                ratio= area/circle_area
                if ratio > THRESH_RATIO_AREA:
                    circles.append(circle)
                    #Check if there exists and overlapping circle already
                    #if overlap(circle,circles): continue
    return circles 


# Create callibrator window
cv2.namedWindow('Callibrator')

# Create trackbars
cv2.createTrackbar('KERNEL_MORPH_DILATE_SIZE','Callibrator',25,100,KERNEL_MORPH_DILATE_SIZE_CHANGE)
cv2.createTrackbar('KERNEL_DILATE_ITER','Callibrator',5,10,KERNEL_DILATE_ITER_CHANGE)
cv2.createTrackbar('CONTOUR_AREA_MIN','Callibrator',70,200,CONTOUR_AREA_MIN_CHANGE)
cv2.createTrackbar('CONTOUR_AREA_MAX','Callibrator',180,200,CONTOUR_AREA_MAX_CHANGE)
cv2.createTrackbar('THRESH_MAX_VALUE','Callibrator',230,255,THRESH_MAX_VALUE_CHANGE)
cv2.createTrackbar('DEPTH_MASK_THRESH','Callibrator',150,255,DEPTH_MASK_THRESH_CHANGE)
cv2.createTrackbar('THRESH_RETAIN_PEAKS','Callibrator',230,255,THRESH_RETAIN_PEAKS_CHANGE)
cv2.createTrackbar('KERNEL_MORPH_OPEN_SIZE','Callibrator',9,19,KERNEL_MORPH_OPEN_SIZE_CHANGE)
cv2.createTrackbar('ROI_START_X','Callibrator',0,720,ROI_START_X_CHANGE)
cv2.createTrackbar('ROI_START_Y','Callibrator',0,1280,ROI_START_Y_CHANGE)
cv2.createTrackbar('ROI_END_X','Callibrator',720,720,ROI_END_X_CHANGE)
cv2.createTrackbar('ROI_END_Y','Callibrator',1280,1280,ROI_END_Y_CHANGE)

# Instantiate the sources
cap =None
pipeline = None
source = 'depth_500.avi'
# get source from args
args = sys.argv
print(args)
# If arguments length is >1
# get the source
if(len(sys.argv)>1):
    
    if source =='cam':
        pipeline = rs.rs.pipeline()
        pipeline.start()
    else :
        #source=sys.argv[1]
        cap = cv2.VideoCapture(source)
    
else:
    cap = cv2.VideoCapture('depth_500.avi')
    
num_frame=0
depth = None

def process(depthx):
    
    out = np.copy(depthx)
    depth =cv2.bilateralFilter(depthx,9, 75, 75)
    final = findHead(255-depth,thresh_rem=THRESH_RETAIN_PEAKS)
    
    #Get all Contours
    cnts = cv2.findContours(final.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #Preprocess the contours
    circles= preprocess(cnts)
    
    for circle in circles:
        ((x,y),r) = circle
        cv2.circle(out, (int(x),int(y)), int(r), (255, 255, 255), 2)
        cv2.circle(final, (int(x),int(y)), int(r), (255, 255, 255), 2)
    cv2.rectangle(final,(ROI_STARTY,ROI_STARTX),(ROI_ENDY,ROI_ENDX),(255,255,255),2)
    cv2.rectangle(out,(ROI_STARTY,ROI_STARTX),(ROI_ENDY,ROI_ENDX),(255,255,255),2)

    final = np.hstack((out,final))
    cv2.line(final,(out.shape[1],0),(out.shape[1],out.shape[0]),(255,255,255))
    cv2.imshow('Callibrator',final)
    
stay = 1
nextFrame =1


while(True):

    #start = time.time()
    #depth, rgb = grabFrame(pipeline)
    if stay == 0 or nextFrame ==1:
        #source - video
        if cap is not None:
            ret,depth = cap.read()
            if ret == 0:
                break
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            
        #source - camera
        else:
            depth, rgb = grabFrame(pipeline)
        nextFrame =0
    
    # Do processing    
    process(depth)
    
    #get keyboard stroke value and process accordingly
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
        
    
    
#Release cap
if cap is not None:
    cap.release()
else:
    pipeline.stop()
configData ={
    
}

#Destroy all windows
cv2.destroyAllWindows()
for i in range (1,30):
    cv2.waitKey(1)

params= {
        "ROI_STARTX":ROI_STARTX,
        "ROI_STARTY":ROI_STARTY,
        "ROI_ENDX":ROI_ENDX,
        "ROI_ENDY":ROI_ENDY, 
        "KERNEL_MORPH_DILATE_SIZE":KERNEL_MORPH_DILATE_SIZE,
        "KERNEL_DILATE_ITER":KERNEL_DILATE_ITER,
        "CONTOUR_AREA_MIN":CONTOUR_AREA_MIN,
        "CONTOUR_AREA_MAX":CONTOUR_AREA_MAX,
        "DEPTH_MASK_THRESH":DEPTH_MASK_THRESH,
        "THRESH_RETAIN_PEAKS":THRESH_RETAIN_PEAKS,
        "THRESH_MAX_VALUE":THRESH_MAX_VALUE,
        "KERNEL_MORPH_OPEN_SIZE":KERNEL_MORPH_OPEN_SIZE,
        "DEPTH_IMAGE_MAX":5000,
        "video_source":"camera",
        "TRACKER_RETAIN_FRAMES":TRACKER_RETAIN_FRAMES,
        "TRACKER_MAX_DIST_THESH":TRACKER_MAX_DIST_THESH,
        "TRACKER_LINE_BOUNDARY":TRACKER_LINE_BOUNDARY,
        "THRESH_RATIO_AREA":THRESH_RATIO_AREA,
        "THRESH_WH_RATIO": THRESH_WH_RATIO

        }

with open('config_data.json', 'w') as f:
    json.dump(params, f)

import requests
import json

url = "https://api.staging.vedalabs.in/v1/rest/behaviours/5d9f06fb6a64eb85bb5b3014"

payload = {
    "name": "person-count-depth",
    "hub": "5d9ef3f3f78f95b45dc6a5d8",
    "camera": "5d9f05276a64eb85bb5b3013",
    "behaviourType": "perosn-count-depth",
    "params": params
}
print(payload)

headers = {
    'Content-type': 'application/json',
    'authorization': "Basic c2F1cmFiaEB2ZWRhbGFicy5pbjojV2FrZXVwQDY="
    }
#pprint(payload)
response = requests.put( url, data=json.dumps(payload), headers=headers)

print(response.text)


# In[ ]:




