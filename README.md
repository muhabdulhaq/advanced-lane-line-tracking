# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify a binary image to "birds-eye view".
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

All code discussed in this writeup can be found in the Advanced-Lane-Line-Tracking notebook in this repository.

## Camera Calibration

Using OpenCV, I used images of a chessboard to create a calibration matrix that could be used to remove distortion from all future images taken with the camera.

Using `cv2.findChessboardCorners`, I found all corners in a 9x6 chessboard. Then, I passed the corners into `cv2.calibrateCamera()` which returns a matrix that can be used in the future with the `cv2.undistort()` method.

Here is an example of a raw image from the camera compared to after undistortion.

![distortion]

## Pipeline

### Gradients

In order to detect high gradient areas of the image (e.g. white or yellow lane lines against pavement), I used the `cv2.Sobel()` method to calculate image derivatives using an extended Sobel operator.

I used the Sobel method in three different ways:

Firstly, to take a directional gradient in either the horizontal or vertical direction. As lane lines are often diagonal from the driver's perspective, these two gradients could be combined to find areas of the image above a certain gradient threshold.

Secondly, to take a magnitude of both directional gradients by squaring both the x and y directions, and then taking the square root of the sum. This provided an additional data point that could be combined with the previous directional gradient.

Lastly, to take a directional gradient and filter based on the angle using the `arctan2()` method. As previously mentioned, lane lines are usually somewhat diagonal. As such, lines that are perfectly horizontal or perfectly vertical can usually be safely discarded.

Using a boolean combination of these three operators, I was able to create a relatively clean binary image isolating the lane lines in the image.

![gradient]

### Color Thresholding

Using HSL and HSV color space, I was able to effectively filter out low saturation and low value pixels, resulting in a binary image with lane lines largely isolated. As you can see, the sky was often also detected, but this was easily cropped out.

![color]

### Gradients + Color Thresholding

I designed both the gradient and the color thresholding methods to be slightly too harsh on their own. I decided that both of them missing parts of the lane line occasionally was better than false positives.

Once they were both finished, I combined the outputs with a boolean 'OR' operation, such that they could complement each others blind spots (under the assumption that they did not share blind spots, which I found to be relatively accurate).

### Perspective Transform

Using the `cv2.getPerspectiveTransform()` method, I created a bird's eye perspective binary image of the road. This allowed for easier detection of lane lines.

A perspective transform shifts works by taking in two sets of four points: a source and a destination. It then "drags" the four source points to the four destination points, modifying the image content as it goes. For example, consider the following image and note the blue (source) and red (destination) points:

![perspective]

To perform a perspective transform, the image will take the four blue points and warp the image as needed to move them to the red points. In this specific instance, this will create a bird's eye view -- although it could be used to a wide variety of interesting applications.

### Lane Line Identification

Once I had generated a binary bird's eye view image with the lane lines relatively isolated, the next step was to simplify the pixel data to two polynomials, one for each lane line. First, I identified the "hottest" (most active pixels in the binary image) area in the bottom left and bottom right quarters of the image. This gave me a good starting point for the search for lane lines. Once this was identified, I used a sliding window approach to continually find the most active area above the already established region, and continue on until I reached the top of the image. This gave me a set of rectangles:

![rect]

Once these rectangles were established, I could use the pixels inside of them to fit two polynomials to, using the very convenient `np.polyfit()` method. In the video, these two polynomials are displayed at all times in the top right of the video, showing the viewer the current simulated bird's eye view.

### Radius of Curvature

With the polynomial created, I used the radius equation: R = (1+(2Ay+B)^2)^(3/2) / abs(2A), where A and B are coefficients of the second order polynomial. As the curvature values may vary slightly between the two lane lines due to inaccuracies in the processing pipeline, I took the average between the two. Certain improvements exist here, such as taking the average of the past 10-20 readings to avoid jumping around excessively.

### Final Result

After identifying the lane lines, plotting them, and calculating the radius of curvature, I then reverted the perspective transform and applied an overlay showing the lane, as well as the current lane curvature and vehicle position.

![final]

---

### Pipeline (video)

Here's a [link to my video result](https://youtu.be/TfRz_oF5x3E)!

---

### Discussion

There are several future areas of development for a system of this nature. Firstly, the perspective transform is computed using hard coded source and destination points. This works well when the road is relatively flat. However, with steep or twisty roads this would not work. A better approach would be to dynamically find the beginning and end of the lane lines in each frame, and then compute the perspective transform based using those points.

Secondly, the gradient and color based features are susceptible to failure in snow, rain, shade, unmarked roads, and more. Perhaps a better approach would be to leverage deep learning techniques.

[distortion]: ./output_images/camera-calibration.png "Undistorted"
[color]: ./output_images/color-space.png "Color Space"
[gradient]: ./output_images/gradient.png "Gradient"
[perspective]: ./output_images/perspective.png "Perspective"
[rect]: ./output_images/rect.png "Rectangles"
[final]: ./output_images/final.png "Final"
