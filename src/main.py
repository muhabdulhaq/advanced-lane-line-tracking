import imageio
imageio.plugins.ffmpeg.download()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
from moviepy.editor import VideoFileClip
import helpers

img_size = (1280, 720) # All images I'm working with have this dimension, this will be a useful variable
w,h = img_size

def process_image(img):
    calibrated = helpers.calibrate_image(img)
    img = helpers.gradient_plus_color_transform(calibrated)
    warped = cv2.warpPerspective(img, helpers.M, img_size, flags=cv2.INTER_LINEAR)
    road, small_road = helpers.analyze_transformed_image(warped)

    # Overlay lane lines with the calibrated image
    road = road.astype(np.uint8)
    road_with_lane_lines = cv2.addWeighted(road, 1.0, calibrated, 0.4, gamma=0.0)
    # Add birds eye view to top right corner (Row,Col coords)
    road_with_lane_lines[0:small_road.shape[0], 1280-small_road.shape[1]:] = small_road

    return road_with_lane_lines

output_file = 'output_video.mp4'
input_clip = VideoFileClip('project_video.mp4').subclip(0,5)
# input_clip = VideoFileClip("project_video.mp4")
output_clip = input_clip.fl_image(process_image) # NOTE: this function expects color images
output_clip.write_videofile(output_file, audio=False)
