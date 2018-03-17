#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import glob
import math
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # recent polynomial coefficients
        self.recent_fit = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # counter to reset after 5 iterations if issues arise
        self.counter = 0

def grayscale(img):
    """Applies the Grayscale transform

    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')

    Args:
        img: a normal image with 3 channels

    Returns:
        a graysacled image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def Camera_Calibrate(row_corners, colum_corners):
    """Using Chessboard to calibrate camera

    In this function, the Camera is calibrated based on the chess board.
    And the intrinsic matrix and Distortion matrix is saved.
    However, the calibration images is setted within the function.

    Args:
        row_corners: row of inner corners
        column_corners: column of inner corners

    Returens:
        the result of calibrated sample image
    """

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((colum_corners * row_corners, 3), np.float32)
    objp[:, :2] = np.mgrid[0:row_corners, 0:colum_corners].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('/home/doublepoints/Documents/selfdriving_T1/CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (row_corners, colum_corners), None)
        #print(ret, "::", corners)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (row_corners, colum_corners), corners, ret)
            # write_name = 'corners_found'+str(idx)+'.jpg'
            # cv2.imwrite(write_name, img)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)

    #cv2.destroyAllWindows()

    import pickle
    # %matplotlib inline

    # Test undistortion on an image
    img = cv2.imread('/home/doublepoints/Documents/selfdriving_T1/CarND-Advanced-Lane-Lines/camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('/home/doublepoints/Documents/selfdriving_T1/CarND-Advanced-Lane-Lines/output_images/undist.jpg', dst)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("./dist_pickle.p", "wb"))
    print(dist_pickle)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # Visualize undistortion
    '''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()
    '''

    return dst

def undistort_image(img, mtx, dist):
    """undistort image

    Use the normal image, intrinsic matrix and distortion matrix to distort an image.

    Args:
        img: distort image
        mtx: intrinsic matrix
        dist: distort matrix

    Returns:
        undistort image

    """

    return cv2.undistort(img, mtx, dist, None, mtx)

def pipeline(img, s_thresh=(125, 255), sx_thresh=(10, 100), R_thresh=(200, 255), sobel_kernel=5):
    """ Pipeline to create binary image.
    This version uses thresholds on the R & S color channels and Sobelx.
    Binary activation occurs where any two of the three are activated.
    """
    distorted_img = np.copy(img)
    dst = cv2.undistort(distorted_img, mtx, dist, None, mtx)
    # Pull R
    R = dst[:, :, 0]

    # Convert to HLS colorspace
    hls = cv2.cvtColor(dst, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobelx - takes the derivate in x, absolute value, then rescale
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

    # Threshold R color channel
    R_binary = np.zeros_like(R)
    R_binary[(R >= R_thresh[0]) & (R <= R_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # If two of the three are activated, activate in the binary image
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) & (sxbinary == 1)) | ((sxbinary == 1) & (R_binary == 1))
                    | ((s_binary == 1) & (R_binary == 1))] = 1

    return combined_binary

def bird_vision(image, image_size, src, dst):
    """Transform image into bird vision

    Args:
        image: the normal image
        image_size: the structure of image
        src: source point(basically 4 points)
        dst: destination points

    Returns:
        perspective image: bird vision image
        M:transform Matrix

    """

    # Compute and apply perpective transform

    M = cv2.getPerspectiveTransform(src, dst)
    perspective_image = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return perspective_image, M

def find_lane(img):
    """Find a lane line[

    Args:
        img:

    Returns:


    """

    binary_warped = img.copy()

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Create an output image to d raw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    #print(window_height)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    # The challenge videos sometimes throw errors, so the below try first
    # Upon the error being thrown, set line.detected to False
    # Left line first
    try:
        n = 5
        left_line.current_fit = np.polyfit(lefty, leftx, 2)
        left_line.all_x = leftx
        left_line.all_y = lefty
        left_line.recent_fit.append(left_line.current_fit)
        if len(left_line.recent_fit) > 1:
            left_line.diffs = (left_line.recent_fit[-2] - left_line.recent_fit[-1]) / left_line.recent_fit[-2]
        left_line.recent_fit = left_line.recent_fit[-n:]
        left_line.best_fit = np.mean(left_line.recent_fit, axis=0)
        left_fit = left_line.current_fit
        left_line.detected = True
        left_line.counter = 0
    except TypeError:
        left_fit = left_line.best_fit
        left_line.detected = False
    except np.linalg.LinAlgError:
        left_fit = left_line.best_fit
        left_line.detected = False

    # Next, right line
    try:
        n = 5
        right_line.current_fit = np.polyfit(righty, rightx, 2)
        right_line.all_x = rightx
        right_line.all_y = righty
        right_line.recent_fit.append(right_line.current_fit)
        if len(right_line.recent_fit) > 1:
            right_line.diffs = (right_line.recent_fit[-2] - right_line.recent_fit[-1]) / right_line.recent_fit[-2]
        right_line.recent_fit = right_line.recent_fit[-n:]
        right_line.best_fit = np.mean(right_line.recent_fit, axis=0)
        right_fit = right_line.current_fit
        right_line.detected = True
        right_line.counter = 0
    except TypeError:
        right_fit = right_line.best_fit
        right_line.detected = False
    except np.linalg.LinAlgError:
        right_fit = right_line.best_fit
        right_line.detected = False

def second_ord_poly(line, val):
    """ Simple function being used to help calculate distance from center.
    Only used within Draw Lines below. Finds the base of the line at the
    bottom of the image.
    """
    a = line[0]
    b = line[1]
    c = line[2]
    formula = (a * val ** 2) + (b * val) + c

    return formula

def count_check(line):
    """ Resets to using new sliding windows below if
    upon failing five times in a row.
    """
    if line.counter >= 5:
        line.detected = False

def lane_detect(image):
    """The main process of processing image

   Args:
       image: the normal image with 3 channels

   Returns:
       result: lane line detected image
   """
    undist_image = undistort_image(image, mtx, dist)  # undistort image

    combied_binary_image = pipeline(undist_image)  # digest a the binary lane image



    #Perspective Transform

    image_size = (image.shape[1], image.shape[0])  # image size without depth
    src = np.float32([[690, 450], [1110, image_size[1]], [175, image_size[1]], [595, 450]])  # Source points - defined area of lane line edges
    offset = 300  # offset for dst points
    dst = np.float32([[image_size[0] - offset, 0], [image_size[0] - offset, image_size[1]],
                      [offset, image_size[1]], [offset, 0]])  # 4 destination points to transfer
    perspective_image, transform_matrix = bird_vision(combied_binary_image, image_size, src, dst)  # transform image into bird vision

    #Find Lane

    # Check if lines were last detected; if not, re-run first_lines
    if left_line.detected == False | right_line.detected == False:
        find_lane(perspective_image)

    # Set the fit as the current fit for now
    left_fit = left_line.current_fit
    right_fit = right_line.current_fit

    # Again, find the lane indicators
    nonzero = perspective_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Set the x and y values of points on each line
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each again.
    # Similar to first_lines, need to try in case of errors
    # Left line first
    try:
        n = 5
        left_line.current_fit = np.polyfit(lefty, leftx, 2)
        left_line.all_x = leftx
        left_line.all_y = lefty
        left_line.recent_fit.append(left_line.current_fit)
        if len(left_line.recent_fit) > 1:
            left_line.diffs = (left_line.recent_fit[-2] - left_line.recent_fit[-1]) / left_line.recent_fit[-2]
        left_line.recent_fit = left_line.recent_fit[-n:]
        left_line.best_fit = np.mean(left_line.recent_fit, axis=0)
        left_fit = left_line.current_fit
        left_line.detected = True
        left_line.counter = 0
    except TypeError:
        left_fit = left_line.best_fit
        count_check(left_line)
    except np.linalg.LinAlgError:
        left_fit = left_line.best_fit
        count_check(left_line)

    # Now right line
    try:
        n = 5
        right_line.current_fit = np.polyfit(righty, rightx, 2)
        right_line.all_x = rightx
        right_line.all_y = righty
        right_line.recent_fit.append(right_line.current_fit)
        if len(right_line.recent_fit) > 1:
            right_line.diffs = (right_line.recent_fit[-2] - right_line.recent_fit[-1]) / right_line.recent_fit[-2]
        right_line.recent_fit = right_line.recent_fit[-n:]
        right_line.best_fit = np.mean(right_line.recent_fit, axis=0)
        right_fit = right_line.current_fit
        right_line.detected = True
        right_line.counter = 0
    except TypeError:
        right_fit = right_line.best_fit
        count_check(right_line)
    except np.linalg.LinAlgError:
        right_fit = right_line.best_fit
        count_check(right_line)

    # Generate x and y values for plotting
    fity = np.linspace(0, perspective_image.shape[0] - 1, perspective_image.shape[0])
    fit_leftx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
    fit_rightx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((perspective_image, perspective_image, perspective_image)) * 255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([fit_leftx - margin, fity]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_leftx + margin, fity])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([fit_rightx - margin, fity]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_rightx + margin, fity])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Calculate the pixel curve radius
    y_eval = np.max(fity)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(left_line.all_y * ym_per_pix, left_line.all_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_line.all_y * ym_per_pix, right_line.all_x * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    avg_rad = round(np.mean([left_curverad, right_curverad]), 0)
    rad_text = "Radius of Curvature = {}(m)".format(avg_rad)

    # Calculating middle of the image, aka where the car camera is
    middle_of_image = image.shape[1] / 2
    car_position = middle_of_image * xm_per_pix

    # Calculating middle of the lane
    left_line_base = second_ord_poly(left_fit_cr, image.shape[0] * ym_per_pix)
    right_line_base = second_ord_poly(right_fit_cr, image.shape[0] * ym_per_pix)
    lane_mid = (left_line_base + right_line_base) / 2

    # Calculate distance from center and list differently based on left or right
    dist_from_center = lane_mid - car_position
    if dist_from_center >= 0:
        center_text = "{} meters left of center".format(round(dist_from_center, 2))
    else:
        center_text = "{} meters right of center".format(round(-dist_from_center, 2))

    # List car's position in relation to middle on the image and radius of curvature
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, center_text, (10, 50), font, 1, (255, 255, 255), 2)
    cv2.putText(image, rad_text, (10, 100), font, 1, (255, 255, 255), 2)

    # Invert the transform matrix from birds_eye (to later make the image back to normal below)
    Minv = np.linalg.inv(transform_matrix)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(perspective_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)


    return result

#TODO:Start the advanced lane detect

#Camera Calibrate
undist_sample_image = Camera_Calibrate(row_corners=9, colum_corners=6)

# Image distortion correction
dist_pickle2 = pickle.load(open("./dist_pickle.p", "rb"))

mtx = dist_pickle2["mtx"]
dist = dist_pickle2["dist"]

# Set the class lines equal to the variables used above
left_line = Line()
right_line = Line()

