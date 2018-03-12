import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import algorithms as alg
import vehicle_tracker as vt

##################################################
# A. Set execution parameters

trainSVM = True
process_images = False
process_videos = True

###################################################
# B. Define parameters for feature extraction

color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
pf = {'color_space':color_space,'orient':orient,'pix_per_cell':pix_per_cell,'cell_per_block':cell_per_block,
'hog_channel':hog_channel,'spatial_size':spatial_size,'hist_bins':hist_bins,'spatial_feat':spatial_feat,
'hist_feat':hist_feat,'hog_feat':hog_feat}

############################################################
# C. Train or Load the SVM

svc,X_scaler,X_test,y_test = alg.getSVM(pf,trainSVM)

############################################################
# D. Test the accuracy of the SVM

print('Test Accuracy of SVC = {}'.format(round(svc.score(X_test, y_test), 4))) # Check the score of the SVC

###########################################################
# E. Run vehicle deteection on test images

import sys
overlap = 0.85
if process_images:


    # image = cv2.imread("output_images/car.png")
    # hogs,hogimage = alg.get_hog_features(image[:,:,0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    # plt.figure(10)
    # plt.imshow(hogimage,cmap=plt.cm.gray)
    # plt.savefig("output_images/hog_car.png")
    
    # image = cv2.imread("output_images/notcar.png")
    # hogs,hogimage = alg.get_hog_features(image[:,:,0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
    # plt.figure(11)
    # plt.imshow(hogimage,cmap=plt.cm.gray)
    # plt.savefig("output_images/hog_notcar.png")
    # plt.show()

    # image = cv2.imread("test_images/test1.jpg")
    # windows = alg.slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 496], xy_window=(72, 72), xy_overlap=(overlap, overlap))
    # windows += alg.slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500], xy_window=(84, 84), xy_overlap=(overlap, overlap))
    # windows += alg.slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 540], xy_window=(96, 96), xy_overlap=(overlap, overlap))
    # windows += alg.slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 580], xy_window=(128, 128), xy_overlap=(overlap, overlap))
    # image = alg.draw_boxes(image, windows, color=(255, 0, 0), thick=2) 
    # plt.figure()
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
    # plt.savefig("output_images/sliding_windows.png")
    # plt.show()

 
    # t0=time.time() 
    # image_names = glob.glob('test_images/test*.jpg')
    # for figctr,image_name in enumerate(image_names):
    #     image = cv2.imread(image_name)
    #     draw_image = np.copy(image)
        
    #     windows = alg.slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 496], xy_window=(72, 72), xy_overlap=(overlap, overlap))
    #     windows += alg.slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 500], xy_window=(84, 84), xy_overlap=(overlap, overlap))
    #     windows += alg.slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 540], xy_window=(96, 96), xy_overlap=(overlap, overlap))
    #     windows += alg.slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 580], xy_window=(128, 128), xy_overlap=(overlap, overlap))
    #     hot_windows = []
    #     hot_windows += (alg.search_windows(image, windows, svc, X_scaler, color_space=color_space, 
    #                         spatial_size=spatial_size, hist_bins=hist_bins, 
    #                         orient=orient, pix_per_cell=pix_per_cell, 
    #                         cell_per_block=cell_per_block, 
    #                         hog_channel=hog_channel, spatial_feat=spatial_feat, 
    #                         hist_feat=hist_feat, hog_feat=hog_feat))       
    #     # draw_image = alg.draw_boxes(draw_image, windows, color=(255, 0, 0), thick=2)                
    #     window_image = alg.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6) 
    #     # plt.subplot(2,3,figctr+1)
    #     plt.figure(figsize=(14,7))
    #     plt.imshow(cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB)) 
    #     plt.savefig("output_images/test%d.png"%figctr)
    #     # plt.show()
    # t1 = time.time()
    # print(round(t1-t0, 2), 'Seconds to process test images')

    vehtracker = vt.VehicleTracker(pf,svc,X_scaler)
    image_names = glob.glob('test_images/test*.jpg')
    for figctr,image_name in enumerate(image_names):
        image = cv2.imread(image_name)
        heatmap = vehtracker.process_frame(image)
        plt.figure()
        plt.imshow(heatmap,cmap=plt.cm.gray)
        plt.savefig('output_images/heat%d.png' % figctr)

    sys.exit()
####################################################
# F. Run the vechicle tracker on a movie stream

if process_videos:
    vehtracker = vt.VehicleTracker(pf,svc,X_scaler)

    vehtracker.makeMovie("project_video.mp4")