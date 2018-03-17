from moviepy.editor import VideoFileClip
from IPython.display import HTML
from vehicle_detect import *
from find_lane import *


def process_image(image):
    """The main process of processing image

   Args:
       image: the normal image with 3 channels

   Returns:
       result: lane line detected image
   """
    # Lane Detection
    pre_result = lane_detect(image)
    # Vehicle Detection
    result = cars_detect(pre_result)

    return result


# Convert to video
# vid_output is where the image will be saved to
vid_output = 'project_vid_output_with_lanes.mp4'

# The file referenced in clip1 is the original video before anything has been done to it
clip1 = VideoFileClip("project_video.mp4")

# NOTE: this function expects color images
vid_clip = clip1.fl_image(process_image)
vid_clip.write_videofile(vid_output, audio=False)