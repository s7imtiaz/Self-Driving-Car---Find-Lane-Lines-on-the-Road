
# Self-Driving Car Engineer Nanodegree


## Project: **Finding Lane Lines on the Road** 
***
In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 

Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.

---
Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.

**Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**

---

**The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**

---

<figure>
 <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
 </figcaption>
</figure>
 <p></p> 
<figure>
 <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
 </figcaption>
</figure>

**Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

## Import Packages


```python
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline
```

## Read in an Image


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
```

    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x111e59d68>




![png](P1_files/P1_6_2.png)


## Ideas for Lane Detection Pipeline

**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**

`cv2.inRange()` for color selection  
`cv2.fillPoly()` for regions selection  
`cv2.line()` to draw lines on an image given endpoints  
`cv2.addWeighted()` to coadd / overlay two images
`cv2.cvtColor()` to grayscale or change color
`cv2.imwrite()` to output images to file  
`cv2.bitwise_and()` to apply a mask to an image

**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

## Helper Functions

Below are some helper functions to help get you started. They should look familiar from the lesson!


```python
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)
```

## Test Images

Build your pipeline to work on the images in the directory "test_images"  
**You should make sure your pipeline works well on these images before you try the videos.**


```python
import os
os.listdir("test_images/")
```




    ['solidYellowCurve.jpg',
     'solidWhiteRight_res.jpg',
     'solidYellowLeft.jpg',
     'solidYellowCurve2_res.jpg',
     'solidYellowCurve_res.jpg',
     'solidYellowCurve2.jpg',
     'solidWhiteCurve_res.jpg',
     'solidYellowLeft_res.jpg',
     'whiteCarLaneSwitch_res.jpg',
     'solidWhiteRight.jpg',
     'whiteCarLaneSwitch.jpg',
     'solidWhiteCurve.jpg']



## Build a Lane Finding Pipeline



Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.

Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.


```python
# TODO: Build your pipeline that will draw lane lines on the test_images

# then save them to the test_images_output directory.

def draw_straight_lines(img, lines, y_min = 315, color = [255, 0, 0], thickness = 16):
    
    #Generate lists to store values
    x1_left = []
    x2_left = []
    y1_left = []
    y2_left = []
    x1_right = []
    x2_right = []
    y1_right = []
    y2_right = []
    
    #Add the lines values to the appropriate list
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1) #get the slope
            if slope >= 0:
                #positive slope, right line
                x1_right.append(x1)
                x2_right.append(x2)
                y1_right.append(y1)
                y2_right.append(y2)
            else:
                #negative slope, left line
                x1_left.append(x1)
                x2_left.append(x2)
                y1_left.append(y1)
                y2_left.append(y2)
    
    #Compute the average value for each point
    x1_left_avg = sum(x1_left) / float (len(x1_left))
    x2_left_avg = sum(x2_left) / float (len(x2_left))
    y1_left_avg = sum(y1_left) / float (len(y1_left))
    y2_left_avg = sum(y2_left) / float (len(y2_left))
    x1_right_avg = sum(x1_right) / float (len(x1_right))
    x2_right_avg = sum(x2_right) / float (len(x2_right))
    y1_right_avg = sum(y1_right) / float (len(y1_right))
    y2_right_avg = sum(y2_right) / float (len(y2_right))
    
    #Get the slope for each side
    left_slope = (y2_left_avg - y1_left_avg) / (x2_left_avg - x1_left_avg)
    right_slope = (y2_right_avg - y1_right_avg) / (x2_right_avg - x1_right_avg)
    
    #Get the intercept for each side
    left_b = y2_left_avg - left_slope * x2_left_avg
    right_b = y2_right_avg - right_slope * x2_right_avg
    
    #Get the y dimension of the image
    y = img.shape[0]
    
    #Calculate the extended lines
    left_y1 = y_min
    left_y2 = y
    left_x1 = int((left_y1 - left_b)/ left_slope)
    left_x2 = int((left_y2 - left_b)/ left_slope)
    right_y1 = y_min
    right_y2 = y
    right_x1 = int((right_y1 - right_b)/ right_slope)
    right_x2 = int((right_y2- right_b)/ right_slope)
    
    #Draw the lines
    cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)
    cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)
    
def hough_straight_lines (img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img,rho,theta,threshold,np.array([]), minLineLength = min_line_len, maxLineGap = max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    draw_straight_lines(line_img, lines)
    return line_img

def process_img(img):
    
    #Parameters
    gaus_kernel = 5 #kernel size for gaussian blur
    canny_low_thres = 50 #Canny low threshold
    canny_high_thres = 150 #Canny high threshold
    mask_vert = np.array([[(0,img.shape[0]), (460,315), (490, 315), (img.shape[1], img.shape[0])]], dtype=np.int32)
    rho = 2
    theta = np.pi/180
    hough_thres = 15
    min_line_len = 40
    max_line_gap = 20
    
    #Process the image and pass return
    gray = grayscale(img)
    blurred = gaussian_blur(gray, gaus_kernel)
    edges = canny(blurred, canny_low_thres, canny_high_thres)
    masked = region_of_interest(edges, mask_vert)
    lines = hough_straight_lines(masked, rho, theta, hough_thres, min_line_len, max_line_gap)
    weighted = weighted_img(lines, img)
    return weighted

def process_dir(dir):
    for img in os.listdir(dir):
        image = mpimg.imread(dir + img)
        res = process_img(image)
        plt.imshow(res)
        img_root = img[:-4]
        plt.savefig(dir + img_root + '_res.jpg')
        
process_dir('test_images/')
    
    
    
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-5-8ae1d0828213> in <module>()
        103         plt.savefig(dir + img_root + '_res.jpg')
        104 
    --> 105 process_dir('test_images/')
        106 
        107 


    <ipython-input-5-8ae1d0828213> in process_dir(dir)
         98     for img in os.listdir(dir):
         99         image = mpimg.imread(dir + img)
    --> 100         res = process_img(image)
        101         plt.imshow(res)
        102         img_root = img[:-4]


    <ipython-input-5-8ae1d0828213> in process_img(img)
         91     edges = canny(blurred, canny_low_thres, canny_high_thres)
         92     masked = region_of_interest(edges, mask_vert)
    ---> 93     lines = hough_straight_lines(masked, rho, theta, hough_thres, min_line_len, max_line_gap)
         94     weighted = weighted_img(lines, img)
         95     return weighted


    <ipython-input-5-8ae1d0828213> in hough_straight_lines(img, rho, theta, threshold, min_line_len, max_line_gap)
         70     lines = cv2.HoughLinesP(img,rho,theta,threshold,np.array([]), minLineLength = min_line_len, maxLineGap = max_line_gap)
         71     line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    ---> 72     draw_straight_lines(line_img, lines)
         73     return line_img
         74 


    <ipython-input-5-8ae1d0828213> in draw_straight_lines(img, lines, y_min, color, thickness)
         16 
         17     #Add the lines values to the appropriate list
    ---> 18     for line in lines:
         19         for x1,y1,x2,y2 in line:
         20             slope = (y2-y1)/(x2-x1) #get the slope


    TypeError: 'NoneType' object is not iterable



![png](P1_files/P1_16_1.png)


## Test on Videos

You know what's cooler than drawing lanes over images? Drawing lanes over video!

We can test our solution on two provided videos:

`solidWhiteRight.mp4`

`solidYellowLeft.mp4`

**Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**

**If you get an error that looks like this:**
```
NeedDownloadError: Need ffmpeg exe. 
You can download it by calling: 
imageio.plugins.ffmpeg.download()
```
**Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**


```python
# Import everything needed to edit/save/watch video clips
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
# Moving average lists and values
x1_left_li = []
x2_left_li = []
x1_right_li = []
x2_right_li = []
x1_left_a = 0
x2_left_a = 0
x1_right_a = 0
x2_right_a = 0

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def mvg_avg_left(x1,x2,qty = 5):
    global x1_left_li, x2_left_li, x1_left_a, x2_left_a
    if len(x1_left_li) > qty:
        x1_left_li.pop(0)
        x2_left_li.pop(0)
    x1_left_li.append(x1)
    x2_left_li.append(x2)
    x1_left_a = int(sum(x1_left_li)/float(len(x1_left_li))) 
    x2_left_a = int(sum(x2_left_li)/float(len(x2_left_li))) 

def mvg_avg_right(x1, x2, qty = 5):
    global x1_right_li, x2_right_li, x1_right_a, x2_right_a
    if len(x1_right_li) > qty:
        x1_right_li.pop(0)
        x2_right_li.pop(0)
    x1_right_li.append(x1)
    x2_right_li.append(x2)
    x1_right_a = int(sum(x1_right_li) / float(len(x1_right_li)))
    x2_right_a = int(sum(x2_right_li) / float(len(x2_right_li)))
    
def draw_straight_lines(img, lines,avg_qty = 5, y_min = 315, color = [255, 0, 0], thickness = 16):
    
    #Generate lists to store values
    x1_left = []
    x2_left = []
    y1_left = []
    y2_left = []
    x1_right = []
    x2_right = []
    y1_right = []
    y2_right = []
    
    #Add the lines values to the appropriate list
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1) #get the slope
            if slope >= 0:
                #positive slope, right line
                x1_right.append(x1)
                x2_right.append(x2)
                y1_right.append(y1)
                y2_right.append(y2)
            else:
                #negative slope, left line
                x1_left.append(x1)
                x2_left.append(x2)
                y1_left.append(y1)
                y2_left.append(y2)
    
    #Compute the average value for each point
    x1_left_avg = sum(x1_left) / float (len(x1_left)) if len(x1_left) != 0 else 0
    x2_left_avg = sum(x2_left) / float (len(x2_left)) if len(x2_left) != 0 else 0
    y1_left_avg = sum(y1_left) / float (len(y1_left)) if len(y1_left) != 0 else 0
    y2_left_avg = sum(y2_left) / float (len(y2_left)) if len(y2_left) != 0 else 0
    x1_right_avg = sum(x1_right) / float (len(x1_right)) if len(x1_right) != 0 else 0
    x2_right_avg = sum(x2_right) / float (len(x2_right)) if len(x2_right) != 0 else 0
    y1_right_avg = sum(y1_right) / float (len(y1_right)) if len(y1_right) != 0 else 0
    y2_right_avg = sum(y2_right) / float (len(y2_right)) if len(y2_right) != 0 else 0
    
    #Get the slope denominator for each side
    left_denom = (x2_left_avg - x1_left_avg)
    right_denom = (x2_right_avg - x1_right_avg)
    
    #Get the x,y dimension of the image
    x = img.shape[1]
    y = img.shape[0]
    
    #Get the slope for each side
    #Get the intercept for each side
    
    if left_denom != 0:
        left_slope = (y2_left_avg - y1_left_avg) / (left_denom)
        left_b = y2_left_avg - left_slope * x2_left_avg
        x1 = int((y_min - left_b) / left_slope)
        x2 = int((y - left_b) / left_slope)
        
        mvg_avg_left(x1,x2,avg_qty)
        
    if right_denom != 0:
        right_slope = (y2_right_avg - y1_right_avg) / (right_denom)
        right_b = y2_right_avg - right_slope * x2_right_avg
        x1 = int((y_min - right_b) / right_slope)
        x2 = int((y - right_b)  /right_slope)
        
        mvg_avg_right(x1,x2, avg_qty)
        
    cv2.line(img, (x1_left_a, y_min), (x2_left_a, y), color, thickness)
    cv2.line(img, (x1_right_a, y_min), (x2_right_a, y), color, thickness)
    
def hough_straight_lines(img,rho,theta,threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), \
                           minLineLength=min_line_len, maxLineGap = max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_straight_lines(line_img, lines)
    return line_img
        
     
    
    
    

def process_image(img):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
     #Parameters
    gaus_kernel = 7 #kernel size for gaussian blur
    canny_low_thres = 50 #Canny low threshold
    canny_high_thres = 150 #Canny high threshold
    mask_vert = np.array([[(0,img.shape[0]), (460,315), (490, 315), (img.shape[1], img.shape[0])]], dtype=np.int32)
    rho = 2
    theta = np.pi/180
    hough_thres = 15
    min_line_len = 55
    max_line_gap = 5
    
    #Process the image and pass return
    gray = grayscale(img)
    blurred = gaussian_blur(gray, gaus_kernel)
    edges = canny(blurred, canny_low_thres, canny_high_thres)
    masked = region_of_interest(edges, mask_vert)
    lines = hough_straight_lines(masked, rho, theta, hough_thres, min_line_len, max_line_gap)
    weighted = weighted_img(lines, img)
    return weighted


    
```

Let's try the one with the solid white lane on the right first ...


```python
white_output = 'solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video solidWhiteRight.mp4
    [MoviePy] Writing video solidWhiteRight.mp4


    100%|█████████▉| 221/222 [00:06<00:00, 35.86it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: solidWhiteRight.mp4 
    
    CPU times: user 2.7 s, sys: 811 ms, total: 3.51 s
    Wall time: 6.77 s


Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```





<video width="960" height="540" controls>
  <source src="solidWhiteRight.mp4">
</video>




## Improve the draw_lines() function

**At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**

**Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

Now for the one with the solid yellow lane on the left. This one's more tricky!


```python
yellow_output = 'solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)
```

    [MoviePy] >>>> Building video solidYellowLeft.mp4
    [MoviePy] Writing video solidYellowLeft.mp4


    100%|█████████▉| 681/682 [00:18<00:00, 35.89it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: solidYellowLeft.mp4 
    
    CPU times: user 8.33 s, sys: 1.9 s, total: 10.2 s
    Wall time: 19.8 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
```





<video width="960" height="540" controls>
  <source src="solidYellowLeft.mp4">
</video>




## Writeup and Submission

If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.


## Optional Challenge

Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!


```python
challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
%time challenge_clip.write_videofile(challenge_output, audio=False)
```


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
```
