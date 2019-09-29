# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2


def circletoRect(circle):
    ((x,y),r)=circle
    return (x-r,y-r,2*r,2*r)

class CentroidTracker():
    def __init__(self, maxDisappeared=5, maxDistance = 50, boundary= 300, skipUpdate= 10):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.trackID = 0
        self.objects = OrderedDict()
        self.objectLocs = OrderedDict()
        self.objectRadius = OrderedDict()
        self.disappeared = OrderedDict()
        self.trackIDs = OrderedDict()
        self.personIn =0
        self.personOut =0
        self.boundary = boundary
        self.skipUpdate = skipUpdate
        self.count = 0

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.multiTracker = cv2.MultiTracker_create()
        self.trackerOn =False
        
    def register(self, centroid,radius,frame):
        # when registering an object we use the next available object
        # ID to store the centroid
        
        self.objects[self.nextObjectID] = centroid
        cx,cy = centroid
        if cy<= self.boundary:
            self.objectLocs[self.nextObjectID] = 0
        else: 
            self.objectLocs[self.nextObjectID] = 1
        
        self.disappeared[self.nextObjectID] = 0
        self.objectRadius[self.nextObjectID]=radius
        self.trackIDs[self.nextObjectID] = self.trackID
        self.trackID += 1
        self.nextObjectID += 1
        self.multiTracker.add(cv2.TrackerCSRT_create(), frame, (cx-radius,cy-radius,2*radius,2*radius))
        
        self.trackerOn=True

    def reInitTracker(self,frame):
        self.trackID =0
        self.multiTracker = cv2.MultiTracker_create()
        objectIDs = list(self.objects.keys())
        for idx in objectIDs:
            x,y=self.objects[idx]
            r= self.objectRadius[idx]
            self.multiTracker.add(cv2.TrackerCSRT_create(), frame, (x-r,y-r,2*r,2*r))
            self.trackIDs[idx] = self.trackID
            self.trackID += 1
            
    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
        
    def overlap(self,circle):
        ((x,y),r)=circle
        objectIDs = list(self.objects.keys())
        for idx in objectIDs:
            x1,y1=self.objects[idx]
            r1= self.objectRadius[idx]
        # check if distance between centres is less than r1 + r2
            if (x1 - x)*(x1 - x) + (y1 - y)*(y1 - y) < (r1+r)*(r1+r):
                #print(x,y,r,x1,y1,r1)
                return True
        return False
        
    def update(self, circles,frame):
        # check to see if the list of input bounding box rectangles
        # is empty
        trackedBoxes = None
        if self.trackerOn==True:
            self.count += 1
            if self.count % self.skipUpdate==0:
                self.reInitTracker(frame)
            success, trackedBoxes = self.multiTracker.update(frame)
            #print(len(list(self.objects.keys())),)
        if len(circles) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects,self.objectRadius,trackedBoxes
        
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(circles), 2), dtype="int")
        inputRadius = np.zeros((len(circles), 1), dtype="int")
        # loop over the bounding box rectangles
        for (i, circle) in enumerate(circles):
            # use the bounding box coordinates to derive the centroid
            ((cX,cY),r) = circle #int((startX + endX) / 2.0)
            #cY = startY #int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputRadius[i]=r
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputRadius[i],frame)
        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
 
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            #print(D)
 
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
 
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                if D[row][col] > self.maxDistance :
                    continue
                
                objectID = objectIDs[row]
                prevLoc = self.objectLocs[objectID]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                _,newCordY = inputCentroids[col]
                
                if prevLoc == 0  and newCordY>self.boundary:
                    self.personOut +=1
                    self.objectLocs[objectID] =1
                if prevLoc == 1 and newCordY<self.boundary:
                    self.personIn +=1
                    self.objectLocs[objectID] =0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                     
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    #Check if the tracker has something
                    trackID= self.trackIDs[objectID]
                    if trackedBoxes is not None:
                        x,y,w,h = trackedBoxes[trackID]
                        cx = x+w/2
                        cy = y+h/2
                        r= (w+h)/2
                        self.objects[objectID] = (cx,cy)                                            
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
               
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            #else:
            for col in unusedCols:
                #Check if this col overlaps with other coord
                if self.overlap((inputCentroids[col],inputRadius[col])):
                    continue
                self.register(inputCentroids[col],inputRadius[col],frame)
 
        # return the set of trackable objects
        return self.objects,self.objectRadius,trackedBoxes
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                