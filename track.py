import numpy as np
import copy

class Track:

    def __init__(self,xy,sxy,num_update_frames=0,frames_since_last_updated=0):
        self.xy = xy  # centroid location in pixels
        self.vxy = np.array([0,0])    #centroid velocity in pixels per frame
        self.sxy = sxy    #box size (half)
        self.num_update_frames = num_update_frames
        self.frames_since_last_updated = frames_since_last_updated
        self.state = 0      #0 uninitiated, 1 ttracked, 2 dead
        self.kappa = 0.2    # 1.0 is all measurement, 0.0 is all track
        self.minupdates_to_initialize = 10

    def predict(self):                  # push position forward by velocity
        self.xy += self.vxy

    def update(self,bxy,bsxy,kappa=None):
        if kappa is None:
            kappa = self.kappa
        # Update the status
        self.frames_since_last_updated = 0 #Get a fresh lease on life
        self.num_update_frames += 1
        if self.num_update_frames>self.minupdates_to_initialize:
            self.state = 1

        # Update the state
        oldxy = copy.copy(self.xy)
        self.xy = kappa*bxy + (1.0-kappa)*self.xy
        self.vxy = (self.xy-oldxy)
        self.sxy = kappa*bsxy + (1.0-kappa)*self.sxy

    def merge(self,tr):
        self.xy = 0.5*(self.xy + tr.xy)
        minx = min(self.xy[0]-self.sxy[0],tr.xy[0]-tr.sxy[0])
        maxx = max(self.xy[0]+self.sxy[0],tr.xy[0]+tr.sxy[0])
        miny = min(self.xy[1]-self.sxy[1],tr.xy[1]-tr.sxy[1])
        maxy = max(self.xy[1]+self.sxy[1],tr.xy[1]+tr.sxy[1])
        self.sxy[0] = 0.5*(maxx-minx)
        self.sxy[1] = 0.5*(maxy-miny)
    
    def get_box(self): # Create box coordinates out of its center and span
        return ( ( int(self.xy[0]-self.sxy[0]) , int(self.xy[1]-self.sxy[1]) ),
            ( int(self.xy[0]+self.sxy[0]) , int(self.xy[1]+self.sxy[1]) ) )