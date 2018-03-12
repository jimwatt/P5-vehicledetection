import numpy as np
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import cv2
import copy
from scipy.ndimage.measurements import label
from scipy.optimize import linear_sum_assignment
import algorithms as alg
from track import Track
from measurement import Measurement


class VehicleTracker:

    def __init__(self,pf,svc,X_scaler,maximum_merge_distance = 32,y_minimum = 440):
        self.pf = pf
        self.svc = svc
        self.X_scaler = X_scaler
        self.maximum_merge_distance = maximum_merge_distance
        self.y_minimum = y_minimum
        self.tracks = []
        self.nonassignment_cost = 250
        self.max_coast_frames = 15

    def process_frame(self,img):

        # First cull any tracks that have not been updated recently
        remove_ind = []
        for tt,track in enumerate(self.tracks):
            if track.frames_since_last_updated>self.max_coast_frames:
                remove_ind.append(tt)
        for ii in sorted(remove_ind, reverse=True):
            del self.tracks[ii]

        # Next, merge tracks that seem very close
        numtracks = len(self.tracks)
        for ii in range(numtracks-1,-1,-1):
            for jj in range(ii):
                dist = np.linalg.norm(self.tracks[ii].xy-self.tracks[jj].xy)
                if(dist<100):
                    self.tracks[jj].merge(self.tracks[ii])
                    del self.tracks[ii]
                    break

        # Predict all tracks forward
        for track in self.tracks:
            track.predict()

        # First, do a search at multiple scales
        hot_boxes = []
        hot_boxes = alg.find_cars(img, 400, 496, 0, 1280, 72./64., 2, self.pf, self.X_scaler,self.svc)
        hot_boxes += alg.find_cars(img, 400, 500, 0, 1280, 84./64., 2, self.pf, self.X_scaler,self.svc)
        hot_boxes += alg.find_cars(img, 400, 540, 0, 1280, 96./64., 2, self.pf, self.X_scaler,self.svc)
        hot_boxes += alg.find_cars(img, 400, 580, 0, 1280, 128./64., 2, self.pf, self.X_scaler,self.svc)

        # Do some extra search around previously identified objects to sniff them out
        for track in self.tracks:
            xy = track.xy
            xi = int(xy[0])
            yi = int(xy[1])
            if(xi>80 and xi<1200 and yi>80 and yi<660):
                hot_boxes += alg.find_cars(img, yi-50, yi+50, xi-50, xi+50, 72./64., 2, self.pf, self.X_scaler,self.svc)
                hot_boxes += alg.find_cars(img, yi-60, yi+60, xi-60, xi+60, 84./64., 2, self.pf, self.X_scaler,self.svc)
                hot_boxes += alg.find_cars(img,  yi-70, yi+70, xi-70, xi+70, 96./64., 2, self.pf, self.X_scaler,self.svc)
                hot_boxes += alg.find_cars(img,  yi-80, yi+80, xi-80, xi+80, 128./64., 2, self.pf, self.X_scaler,self.svc)
        
        # Add up all the heat and compute labels
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = alg.accumulate_heat(heat,hot_boxes)
        heat = alg.apply_threshold(heat,3)
        heatmap = np.array(np.clip(heat, 0, 255),dtype=np.float32)

        labels = label(heatmap)
        cars_boxes = self.update_tracks(labels)
        imp = alg.draw_boxes(np.copy(img), cars_boxes, color=(0, 0, 255), thick=6)

        return heatmap*255/np.amax(heatmap)

    def update_tracks(self,labels):

        # Given the current labels, update the track estimates and return bounding boxes
        
        # Format the measurements
        measurements = []
        meas_boxes = []
        for car_number in range(1, labels[1]+1):
            nonzero = (labels[0] == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            meas_boxes.append(bbox)
            size_x = 0.5*(bbox[1][0]-bbox[0][0]) 
            size_y = 0.5*(bbox[1][1]-bbox[0][1])
            x = bbox[0][0] + size_x
            y = bbox[0][1] + size_y
            size_xa,size_ya = alg.adjust_aspect_ratio(size_x,size_y)
            measurements.append(Measurement(x,y,size_xa,size_ya))
        nummeasurements = len(measurements)
        
        # If we actually got any measurements
        if(nummeasurements>0):
            numtracks = len(self.tracks)
            # If we already have at least one track
            if(numtracks>0):
                cost_matrix = np.zeros((numtracks+nummeasurements,nummeasurements))
                for tt in range(numtracks):
                    txy = self.tracks[tt].xy
                    for mm in range(nummeasurements):
                        mxy = measurements[mm].xy
                        cost_matrix[tt,mm] = np.linalg.norm(mxy - txy)

                cost_matrix[numtracks:,:] = self.nonassignment_cost


                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                updated_tracks = set()
                for rr,cc in zip(row_ind,col_ind):
                    if rr<numtracks:
                        # Update the track
                        self.tracks[rr].update(measurements[cc].xy,measurements[cc].sxy)
                        updated_tracks.add(rr)
                    else:
                        # Inititate a new track
                        self.tracks.append(Track(measurements[cc].xy,measurements[cc].sxy))
                for uu in range(len(self.tracks)):
                    if uu not in updated_tracks:
                        self.tracks[uu].frames_since_last_updated += 1

            # Else, these measurements are all new tracks 
            else:
                for meas in measurements:
                    self.tracks.append(Track(meas.xy,meas.sxy))

        # generate boxes for the tracks
        boxes = []
        for tracki in self.tracks:
            if(tracki.state>0):
                nextbox = tracki.get_box()
                boxes.append(nextbox)
        return boxes

    # def process_image(self,image):
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     return cv2.cvtColor(self.process_frame(image), cv2.COLOR_BGR2RGB)


    def process_image(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        newimage = self.process_frame(image)
        return cv2.cvtColor(newimage,cv2.COLOR_GRAY2RGB)


    def makeMovie(self,moviename):
        output_v = "test_{}".format(moviename)
        clip1 = VideoFileClip(moviename)
        clip = clip1.fl_image(self.process_image).subclip(40,45)
        clip.write_videofile(output_v, audio=False)