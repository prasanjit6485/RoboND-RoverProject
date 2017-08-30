## Project: Search and Sample Return

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./test_dataset/obstaclesample.jpg
[image3]: ./test_dataset/rocksample.jpg
[image4]: ./test_dataset/perspective_transform.png
[image5]: ./test_dataset/color_thresholding.png

![alt text][image1]

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

I am submitting the writeup as markdown and You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

Here I have considered two images one for obstacle sample and rock sample as mentioned below and performed basic image processing like perpective transform, color thresholding and morphological operation as first step in perception algorithm.

Rock sample image

![alt text][image3]

Obstacle sample image

![alt text][image2]

In Perspective tranform step, I have used cv2 getPerspectiveTransform and warpperspective function as mentioned in the lesson for mapping purpose. Also, I have created mask image of Rover's field of view which will be later used for obstacle pixels in color thresholding step.

Perspective transform (top grid: Rock sample image, bottom grid: Obstacle sample image)

![alt text][image4]

In Color thresholding step, I have used basic thresholding operation to identify navigable pixels as well as rock sample pixels. Also, I have used cv2 inRange and bitwise_and to identify rock sample pixels. To enhance rock sample pixels, I have perfomed dilation morphological operation. For obstacle pixels, I have used warped mask image and subtract from navigable threshold image. Below we see for rock sample image (2nd and 3rd column), we are able to extract rock pixels after performing color thresholding. First column represents navigable terrain pixels and last column represents obstacle pixels in white.

Color thresholding (top grid: Rock sample image, bottom grid: Obstacle sample image)

![alt text][image5]

#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

In process_image() function, following steps I have performed to map pixels to idetify navigable terrain and obstacles into a worldmap and create video using moviepy function.

1) Define source and destination points for perspective transform
2) Apply perspective transform for mapping purpose and create warped mask image
3) Apply color threshold to identify navigable terrain/obstacles/rock samples
4) Convert thresholded image pixel values to rover-centric coords
5) Convert rover-centric pixel values to world coords
6) Update worldmap (to be displayed on left side of video)
7) Create output image consisting of original image, warped image and worldmap
8) Finally create video using moviepy function

![Output sample](./output/jupyter_mapping.gif)

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

I have posted a video online https://www.youtube.com/watch?v=x8kZY2MCl70 where Rover search and pick-up all rock samples and return home successfully. I am running Roversim on MacOS with screen resolution as 1024x768 and graphic quality as Good. 

Following are my conclusion with rover launching in autonomous mode:
1) Rover was able to navigate effectively in the direction of rock samples.
2) Rover was able to induce 4-wheel turn effectively.
3) Rover was able to come out of stuck state near wall or near obstacle or even sometime while collecting rock sample by incorporating 45 degree turn when rover is stuck at one place for more than 5 sec.
4) Rover was able to improve fidelity by updating map when Rover's roll and pitch is within 0.5 degree.
5) Rover was able to detect and avoid obstacles effectively by limiting the Rover's field of view.
6) Rover was effectively crawling towards right side of the wall to collect rock samples and avoid circular motion when navigable terrain is very wide.
7) Rover was effectively avoid wall crawling when navigable terrain is very narrow to avoid bumps and improve fidelity.
8) Rover was able to return home when Rover is within 5m of radius from home location.

Following are my conclusions where I can improve the code:
1) Rover can avoid the area where it has already mapped successfully and collected all rock samples.
2) Rover is unable to visit hidden places due to lack of light condition. Instead of considering RGB model, I can process the image in HSV model.
3) After couple of testing, I found out there is one place near home location where Rover moves in circular motion (even after wall crawling implementation) and was not able to exit the circular motion. Need to implement fitting based algorithm/RANSAC algorithm to collect all coordinates and identify whether Rover is moving in circular motion.
4) Also, need to implement path finding algorithm (A star search algorithm) to return home and collect rock samples.
