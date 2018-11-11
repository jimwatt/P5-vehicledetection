# Vehicle Detection and Tracking

![alt text][image20]

This project is in conjunction with the Udacity Self-Driving Car course.  In particular, the goal of this project is to train a SVM to detect vehicles from video imagery.  Given an input video of the road ahead, this code will track the vehicles in the video and annotate them with bounding boxes.

* More detail about the lane finding approach is provided in writeup.md.

## Getting Started

### Prerequisites

Imported packages include:

```
Python
cv2
matplotlib
moviepy
glob
```

### Installing

No install is required --- simply clone this project from GitHub:

```
git clone https://github.com/jimwatt/P5-vehicledetection.git
```

## Running the Code

* The starting point for the code is `p5code.py`.  In the top directory, run 

  `ipython p5code.py`

  Consider changing flags in main() to switch between processing video and still images.





## Authors

* **James Watt**

<!--## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details-->

## Acknowledgments
This project is a submission to the Udacity Self-Driving Car nanodegree:

* ```
  https://github.com/udacity/CarND-Vehicle-Detection.git
  ```

## Vehicle Detection and Tracking

------

**Vehicle Detection Project**

The goals / steps of this project are the following:

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
- Also apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
- Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
- Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.

[//]: #	"Image References"
[image1]: ./output_images/car.png
[image2]: ./output_images/notcar.png
[image3]: ./output_images/hog_car.png
[image4]: ./output_images/hog_notcar.png
[image5]: ./output_images/sliding_windows.png
[image6]: ./output_images/test0.png
[image7]: ./output_images/test1.png
[image8]: ./output_images/test2.png
[image9]: ./output_images/test3.png
[image10]: ./output_images/test4.png
[image11]: ./output_images/test5.png
[image12]: ./output_images/framesC/C001.png
[image13]: ./output_images/framesC/C002.png
[image14]: ./output_images/framesC/C003.png
[image15]: ./output_images/framesC/C004.png
[image16]: ./output_images/framesB/B001.png
[image17]: ./output_images/framesB/B002.png
[image18]: ./output_images/framesB/B003.png
[image19]: ./output_images/framesB/B004.png
[image20]: ./output_images/framesA/A001.png
[image21]: ./output_images/framesA/A002.png
[image22]: ./output_images/framesA/A003.png
[image23]: ./output_images/framesA/A004.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

------

### Writeup 

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.    

You're reading it!

The main entry point for the python code is the file `p5code.py`.

Run the code by running `ipython p5code.py`.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 37-49 of the file called `algorithms.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

![alt text][image2]

Also, note that I augmented the image data set by including flipped versions of each image (about the y-axis).

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

**Car**

![alt text][image3]

**Not Car**

![alt text][image4]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters.  To be honest, I cannot say that this was a very scientific endeavor.  I tried to use the grid search as recommended, but even this was somewhat doubtful.  I wasn't too concerned about this choice because it seemed that I could basically enter anything reasonable and still get somewhat good results.    

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in line 66 of `algorithms.py`.  The linear classifier achieved an accuracy of 0.9875 on the test data.  

I did find that although I achieved good performance on the test data, this did not generalize very well to the images from the movie streams.  Incorporating more data similar to the movie imagery would have been helpful here.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I applied various scalings and translations to find good coverage for the sliding windows.  I decided to use four different windows sizes to capture cars both close and far.  This is implemented in lines 50-54 in `vehicle_tracker.py ` .  An example coverage is shown here:

![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

## ![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]![alt text][image10]

![alt text][image11]

Certainly, there are false positives in these images that I addressed using a tracking approach.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./annotated_project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I treated this problem as fundamentally a _tracking_ problem, rather than just a *detection* and update problem.  

Consequently, I extended the suggested approach of using heatmaps beyond just updating and labeling the heatmap.  

My approach included the following main principles:

- I used a filtering approach to estimate the dynamic state of the vehicles in the imagery.
- A track was initiated for each new vehicle detected.  The track object is defined in `track.py`
- The state to be estimated for each track included the following quantities:
  - `(x,y)` The location of the vehicle in pixels
  - `(vx,vy)` The velocity of the vehicle in pixels per frame
  - `(sx,sy)` The size of the bounding box for the vehicle
- For each new image frame, I generated a heatmap using the detection windows in the conventional way.
- Then, I used the labeling approach to generate bounding boxes.
- These bounding boxes, were then used as _measurements_ for the filter.  The measurement object is defined in `measurement.py`.
- First, I solved the measurement-track assignment problem using `scipy.optimize.linear_sum_assignment ` .  
- The costs in the assignment are simply the distance from the measurement to the track.
- The assignment also allowed for no assignment, so that new tracks can be initiated, and spurious measurements can be thrown away.
- After being updated by measurements for several frames, tracks are graduated to full track status (initiated).
- Prediction of track location is performed using simple NCV model.
- Track states are updated with the measurements.  
- If a track does not receive any measurement updates for several frames it is deleted.
- Tracks can also be merged if they become too close to eachother.
- The fundamental state recorded was not the heat map (that is treated as a measurement), but rather the location, velocity, and shape of the vehicles.

The tracking system can be found in `vehicle_tracker.py`.

The intent of the tracking system is to allow smoother tracks to be generated from the noisy detection measurements.  This is in keeping with a true tracking and estimation system, rather than _ad hoc_ heat map approaches that simply try to smooth the measurements rather than estimating the true underlying state.

I liked this approach because it worked "out of the box".  I coded it with reasonable parameters, and it did not require any tuning.  

### Here are four frames of heatmaps:

**Heat Maps**

![alt text][image12]

![alt text][image13]



![alt text][image14]

![alt text][image15]

**Bounding Boxes returned from "labels" function.**

![alt text][image16]

![alt text][image17]

![alt text][image18]

![alt text][image19]

**Bounding Boxes Returned from the Tracker**

![alt text][image20]

![alt text][image21]

![alt text][image22]

![alt text][image23]



- From these image series it is plain to see that detection is somewhat sporadic and noisy.  However, through filtering and tracking we can obtain smooth tracks on the vehicles.

------

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found that for the given training sets, the classifier could work well on the training and test data (+98% accuracy), but was not as robust in the video imagery.

Consequently, the vehicle detection was somewhat sporadic and noisy.

Expanding the training data to include more imagery representative of the videos could help alleviate this.

I opted for a tracking approach in which I update the track state of the vehicles (that are smoothly varying quantities) with the noisy detection measurements (bounding boxes from the heat maps).

The filtering approach worked well as evidenced by the final annotated video.

