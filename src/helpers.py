from tracker import Tracker
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

PATH = Path('../camera_cal/')
n_cols = 9
n_rows = 6
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

img_size = (720, 1280) # All images I'm working with have this dimension, this will be a useful variable
h,w = img_size

objp = np.zeros((n_cols*n_rows,3), np.float32)
objp[:,:2] = np.mgrid[0:n_cols,0:n_rows].T.reshape(-1,2)

for filename in list(PATH.iterdir()):
    img = plt.imread(filename)
    if img is None:
        print('ERROR')
        pdb.set_trace()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (n_cols, n_rows), None)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        img = cv2.drawChessboardCorners(img, (n_cols, n_rows), corners, ret)
    else:
        print('Unable to calibrate', filename)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

offset = w*.25
dst = np.float32(
[[w*.25,0], # Top left
[w*.75, 0], # Top right
[w*.25, h], # Bottom left
[w*.75, h]]) # Bottom right

## Stealing from youtube...
mid_width = 0.08
bot_width = 0.76
height_pct = .62
bottom_trim = 0.935

top_left =  (w*(0.5-mid_width/2), h*height_pct)
top_right = (w*(0.5+mid_width/2), h*height_pct)
bottom_left =  (w*(0.5-bot_width/2), h*bottom_trim)
bottom_right = (w*(0.5+bot_width/2), h*bottom_trim)

src = np.float32([top_left, top_right, bottom_left, bottom_right])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

def calibrate_image(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Using mpimg.imread
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
       sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
       sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    max_sobel = np.max(abs_sobel)
    scaled_sobel = abs_sobel * 255 / max_sobel
    scaled_sobel = scaled_sobel.astype(np.uint8)
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    s_binary = np.zeros_like(scaled_sobel)
    s_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
#     print(np.max(scaled_sobel))
    return s_binary

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Using mpimg.imread
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # x gradient
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # y gradient

    # Calculate the magnitude
    sum_sq = np.sqrt(sobel_x**2 + sobel_y**2)
    max_sobel = np.max(sum_sq)

    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = sum_sq * 255 / max_sobel
    scaled_sobel = scaled_sobel.astype(np.uint8)

    # Create a binary mask where mag thresholds are met
    post_threshold = np.zeros_like(scaled_sobel)
    post_threshold[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return post_threshold

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Using mpimg.imread
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # x gradient
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # y gradient
    # Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    # Use arctan2 to calculate the direction of the gradient
    atan = np.arctan2(abs_sobel_y, abs_sobel_x) # Element-wise arc tangent of y/x
    # Create a binary mask where direction thresholds are met
    post_threshold = np.zeros_like(atan)
    post_threshold[(atan >= thresh[0]) & (atan <= thresh[1])] = 1
    return post_threshold

### Parameters to tune
ksize = 7 # Choose a larger odd number to smooth gradient measurements
thresh = (50, 255)
mag_thresh = (50, 255)
dir_thresh = (0.8, 1.2)

def gradient_sobel(img):
    ''' Takes an RGB image and computes several gradients to remove non-lane pixels
    '''
    # Apply each of the thresholding functions
    gradx = abs_sobel_threshold(img, orient='x', sobel_kernel=ksize, thresh=thresh)
    grady = abs_sobel_threshold(img, orient='y', sobel_kernel=ksize, thresh=thresh)
    mag_binary = mag_threshold(img, sobel_kernel=ksize, mag_thresh=mag_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=dir_thresh)
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined

### Parameters to tune
s_min = 200
s_max = 255
v_min = 215
v_max = 255

def color_transform(img):
    ''' Takes an RGB image, converts it to HSV and HLS, and uses thresholds to remove non-lane pixels
    '''
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_min) & (s_channel < s_max)] = 1

    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > v_min) & (v_channel < v_max)] = 1

    out = np.zeros_like(s_channel)
    out[(s_binary == 1) | (v_binary == 1)] = 1
    return out

def gradient_plus_color_transform(img):
    ''' Converts an RGB image to a bimary lane line image using gradients + color space transforms
    '''
    grad = gradient_sobel(img)
    color = color_transform(img)

    img = np.zeros_like(color)
    img[(grad == 1) | (color == 1)] = 255 # At the end of the pipeline, create an image that's 0-255 scaled
    return img

def window_mask(width, height, img_ref, center, level):
    ''' Creates a mask of 1's
    '''
    h,w = img_ref.shape
    output = np.zeros_like(img_ref)
    output[int(h-(level+1)*height):int(h-level*height),max(0,int(center-width)):min(int(center+width),w)] = 1
    return output

window_width=25
window_height=80
xm_per_pix=4/384
ym_per_pix=10/720

def analyze_transformed_image(img):
    ''' Takes in a bird's eye view single channel image of lane lines, and calculates their position and curvature
    '''
    curve_centers = Tracker(window_width=window_width, window_height=window_height, margin=25,
                            ym=ym_per_pix, xm=xm_per_pix, smooth_factor=15)
    warped = img
    window_centroids = curve_centers.find_window_centroids(warped)

    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    leftx = []
    rightx = []

    for level in range(0, len(window_centroids)):
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0],level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1],level)

        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])

        # Save the boxes for the next run
        l_points [(l_points == 255) | (l_mask == 1)] = 255
        r_points [(r_points == 255) | (r_mask == 1)] = 255

    template = np.array(r_points+l_points, np.uint8) # merge left and right window pixels
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel,template,zero_channel)), np.uint8)
    warpage = np.array(cv2.merge((warped,warped,warped)), np.uint8)
    result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

    yvals = range(0,h)
    res_yvals = np.arange(h-window_width/2,0,-window_height)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx,np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = left_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx,np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    # middle_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

    # road = np.zeros_like(original_img)
    road = np.zeros((h, w, 3))
    # road_bkg = np.zeros_like(img)
    # print(len(left_lane))
    # print(road.shape)
    # print(img.shape)
    cv2.fillPoly(road, [left_lane], color=[255,0,0]) # Left is red
    cv2.fillPoly(road, [right_lane], color=[0,0,255]) # Right is blue
    # cv2.fillPoly(road_bkg, left_lane,color=[255,255,255])
    # cv2.fillPoly(road_bkg, right_lane,color=[255,255,255])

    ## Determine where the car sits inside the lane: to the right or left and by how much
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center - w/2) * xm_per_pix
    side_pos = 'left' if center_diff < 0 else 'right'

    ## Determine radius of curvature
    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix,np.array(leftx,np.float32)*xm_per_pix,2)
    # Determine the curvature of left lane
    curve_rad = ((1+(2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])

    # Warp from birds eye to camera perspective
    camera_perspective = cv2.warpPerspective(road, Minv, (w,h), flags=cv2.INTER_LINEAR)

    cv2.putText(camera_perspective, 'The Radius of Curvature = ' + str(round(curve_rad,3))+'(m)',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(camera_perspective, 'The Vehicle is ' + str(abs(round(center_diff,3)))+'m ' + side_pos + ' of center',(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    small_birds_eye = cv2.resize(road, (0,0), fx=0.3, fy=0.3)
    # print('In analyze_transformed_image')
    # print('road.shape = ', camera_perspective.shape)
    # print('Small road.shape = ', small_birds_eye.shape)

    return camera_perspective, small_birds_eye # road, small_road
