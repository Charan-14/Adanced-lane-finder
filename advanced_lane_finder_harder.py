## Importing Required Libraries

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
np.set_printoptions(precision=3, suppress=True)
### Pipeline
 
# 1. Computing the camera calibration matrix and distortion coefficients given a set of chessboard images to undistort raw images.
# 2. Using HLS colour transform and Sobelx gradient finding filter to create a thresholded binary image.
# 3. Applying perspective transform to have a birds eye view for lane finding.
# 4. Detecting lane pixels and fiting a line to find the lane boundary.
# 5. Determinig the curvature of the lane and vehicle position with respect to center.
# 6. Warping the detected lane boundaries back onto the original image.
# 7. Visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Functions 
# 1. Calibrating camera with chessboard images to find calibration matrix and distortion coefficients which are used to undistort raw images

def calibrateCam(): 
# Importing chessboard images for finding the corners to compute correction matrix    
    path = 'camera_cal'
    images = os.listdir(path)
    
    obj_pts = []
    img_pts = []
# Set of real pts in real world space
    obj_pts_gen = np.zeros((9*6, 3), np.float32)
    obj_pts_gen[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Finding actual image space corner pts from each chessboard image 
    for img in images:
        cal_img = cv.imread(path + '/' + img)

        gray = cv.cvtColor(cal_img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (9,6), None)

        if ret==True:
            img_pts.append(corners)
            obj_pts.append(obj_pts_gen)
            # Visualizing img poinrs to see the accuracy of point detection
            cal_img = cv.drawChessboardCorners(cal_img, (9,6), corners, ret)
            
    # This function gives us the camera matrix and distortion coefficients required for undistorting an image  
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_pts, img_pts, gray.shape[::-1], None, None)
    
    return mtx, dist

# 2. Using HLS Colour Space and Sobel Gradient Computation to isolate lane lines in all lighting conditions and create a binary thresholded image with undistortion

def colorGradientThreshold(frame, mtx, dist):

    # Calculates the direction of of the gradient
    def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
        
        # Calculates the x and y gradients, takes in grayscale image
        sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # Takes the absolute value of the gradient direction,  applys a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output    

    # Calculates the magnitude of the gradient in both direction x and y

    def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

        # Take both Sobel x and y gradients, uses a grayscale image 
        sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # Calculates the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)

        # Scales the value
        scale_factor = np.max(gradmag)/255 
        gradmag = gradmag/scale_factor

        # Creates a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
        return binary_output

    def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
        
        # Applies x or y gradient with the OpenCV Sobel() function and takes the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv.Sobel(img, cv.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv.Sobel(img, cv.CV_64F, 0, 1))
        
        # Scaling for better results
        scaled_sobel = 255*abs_sobel/np.max(abs_sobel)

        # Creates a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Creates a binary image which meets the threshold values
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
        return binary_output

    # Using camera matrix and distortion coeff to undistort each frame of the video
    undist = cv.undistort(frame, mtx, dist, None, mtx)
    
    # Using Gaussian Blur to smoothen images
    kernel = 3 
    gaussBlur = cv.GaussianBlur(undist, (kernel,kernel), 0)
    
    # Using different color spaces for better detection of lanes
    lab = cv.cvtColor(gaussBlur, cv.COLOR_BGR2LAB)
    hls = cv.cvtColor(gaussBlur, cv.COLOR_BGR2HLS)
    gray = cv.cvtColor(gaussBlur, cv.COLOR_BGR2GRAY)
    
    # Using lightness to isolate white lane by HLS color space
    lower_hls = np.array([0, 247, 230]) 
    upper_hls = np.array([255, 255, 255]) 
    mask_hls = cv.inRange(hls, lower_hls, upper_hls)
    res_hls = cv.bitwise_and(gaussBlur, gaussBlur, mask=mask_hls)

    # Using B*(from blue to yellow) in LAB color space to isolate yellow lane
    lower_lab = np.array([182, 100, 156])
    upper_lab = np.array([255, 255, 255])
    mask_lab = cv.inRange(lab, lower_lab, upper_lab)
    res_lab = cv.bitwise_and(gaussBlur, gaussBlur, mask=mask_lab)

    # Combining the result of both the color spaces
    color_comb = cv.bitwise_or(res_lab,res_hls)
    color_comb = cv.cvtColor(color_comb, cv.COLOR_BGR2GRAY)

    # Converting to a binary image
    ret, color_comb = cv.threshold(color_comb,0,255,0)

    gradx = abs_sobel_thresh(gray, orient='x', thresh_min=20, thresh_max=100)
    combined = cv.bitwise_or(gradx, np.float64(color_comb))

    combined_binary = combined

    # Returning color threshold image and undistorted image
    return combined_binary, undist

# 3. Perpective Transform for a birds eye view at the lanes to easily find the curve in the lanes.

def perspectiveTransform(combined_binary): 
    # Source vertices
    vertices = [(460,520), (790,520), (1180,720), (170, 720)]
    src = np.float32(vertices)

    img_size = combined_binary.shape

    # Destination vertices
    dst = np.float32([(180,0), (img_size[1]-350,0), (img_size[1]-350,img_size[0]), (150, img_size[0])])
    
    # For visualization of source vertices
    ver = np.array(([vertices]), np.int32)
    #cv.polylines(undist, ver, True, (0,0,255), thickness=3)

    # Finding Transformation and inverse transformation matrix to warp or unwarp the image
    Mat = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src)
    
    # Warping into a new top angle pesrpective using the transformation matrix
    warped_binary = cv.warpPerspective(combined_binary, Mat, (combined_binary.shape[1], combined_binary.shape[0]), flags=cv.INTER_LINEAR)
    
    return warped_binary, Minv


# 4. Using Histogram to find maximum pixel concentration area to start line fitting

def slideWindow(warped_binary):
    global prev_left_inds
    global prev_right_inds

    # creating a histogram with y axis as the sum of the pixels and the x axis as the x coordinates of the image
    histogram = np.sum(warped_binary[warped_binary.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)
    
    out_img = np.dstack((warped_binary, warped_binary, warped_binary))
    
    # Finding the midpoint of the histogram
    midpoint = np.int(histogram.shape[0]//2)

    # Finding the x coordinate of the image which has the highest pixel sum in the left half of the image    
    leftx_base = np.argmax(histogram[:midpoint])    

    # Finding the x coordinate of the image which has the highest pixel sum in the right half of the image
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # The number of sliding windows
    nwindows = 9
    
    # Width of the windows +/- margin
    margin = 60
    
    # Minimum number of pixels found to recenter window
    minpix = 50

    # Setting height of windows - based on nwindows above and image shape
    window_height = np.int(warped_binary.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    nonzero = warped_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_binary.shape[0] - (window+1)*window_height
        win_y_high = warped_binary.shape[0] - window*window_height
        
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin 
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identifing the nonzero pixels in x and y within the window 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
    
        # If you found greater pixels than minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
    # Avoids an error if the above is not implemented fully
        pass
    
    # Extract left and right line pixel positions
    
    if len(left_lane_inds)!=0:
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        prev_left_inds = left_lane_inds
        #print('l', prev_left_inds)
    else:
        #print('l', prev_left_inds)
        leftx = nonzerox[prev_left_inds]
        lefty = nonzeroy[prev_left_inds]
        left_lane_inds = prev_left_inds
        
    if len(right_lane_inds)!=0:
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        prev_right_inds = right_lane_inds
        #print('r', prev_right_inds)  
    else:
        #print('r', prev_right_inds)
        rightx = nonzerox[prev_right_inds]
        righty = nonzeroy[prev_right_inds]
        right_lane_inds = prev_right_inds
    
      
    # Returning line pixels and lane indices
    return leftx, lefty, rightx, righty, out_img, left_lane_inds, right_lane_inds

# Fitting a polynomial on these line pixels
def fit_polynomial_window(warped_binary):

    # Finding lane line pixels using sliding window method
    leftx, lefty, rightx, righty, out_img, left_lane_inds, right_lane_inds = slideWindow(warped_binary)

    # Fitting a second order polynomial to both lane line pixels using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generating y values for plotting
    ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0])

    # Using the coefficeints from polyfit and the y axis values we fit find a polynomial 
    # x = Ay**2 + By + C
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    # Returning output image, Polynomial x and y values, Coeffcients of polynomial curve and lane indices 
    return out_img, left_fitx, right_fitx, ploty, left_fit, right_fit, left_lane_inds, right_lane_inds

# 5. Using previous polynomial coefficients to find lane line around a certain margin instead of blind search

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    
    # Fitting a second order polynomial to both lane line pixels using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])

    # Calculating x value for fitted polynomial
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty, left_fit, right_fit

def search_around_poly(warped_binary, left_fit, right_fit):

    # HYPERPARAMETER
    # The width of the margin to search aroud the previous polynomial
    
    margin = 50

    # Grab activated pixels
    nonzero = warped_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Indices around the +/- a margin around region of previous polynomial 
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(warped_binary.shape, leftx, lefty, rightx, righty)
    
    
    ## Visualization ##
    # Creating an image to draw on and an image to show the selection window
    out_img = np.dstack((warped_binary, warped_binary, warped_binary))*255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generating a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    
    return result, left_fitx, right_fitx, ploty, left_fit, right_fit, left_lane_inds, right_lane_inds

# Unwarps the birds eye view back to the original image
def warpBack(warped_binary, left_fit, right_fit, ploty):
    
    # Creating an image to draw the lines on
    warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ###### X value of polynomial to find the boundaries of the lanes
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 

    # Combine the result with the original image
    unwarped = cv.addWeighted(undist, 1, newwarp, 0.4, 0)

    # Return the unwarped image
    return unwarped

# 6. Calculating Radius of Curvature and Car's postion with respect to the middle of the lane.
def measure_curvature_pixels(warped_binary, leftx, lefty, rightx, righty):

    # Length of real world space lane which comes in one frame
    ym_per_pix = 30/720 # meters per pixel in y dimension
    
    # Width of lane in real world space
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit a second order polynomial to each using `np.polyfit` and converting to meters
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*ym_per_pix, 2)

    # Generate y values for plotting
    ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0])
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Using Radius of Curvature formula to calculate the value
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    curve_rad = (left_curverad + right_curverad)/2

    return curve_rad

def carPosition(unwarped, left_fit, right_fit):
    
    # Height of image
    height = unwarped.shape[0]
    
    # Middle of car is the middle of the image
    car_middle = unwarped.shape[1]/2
    
    # Lane Polynomial x value
    left_fitx = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
    right_fitx = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
    
    # Center of Lane
    lane_centerx = (left_fitx+right_fitx)/2

    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Car's position with respect to the center of the lane
    car_position = (lane_centerx-car_middle)*xm_per_pix
    
    if car_position>0:
        pos_text = "right of center by " + '{:03.2f}'.format(abs(car_position))
    else:
        pos_text = "left of center by " + '{:03.2f}'.format(abs(car_position))
    return pos_text    

# Displaying Curvature and Car's position info on top of the image
def display(unwarped, curve_rad, pos_text):

    font = cv.FONT_HERSHEY_SIMPLEX
    
    rad_text = 'Radius of Curvature of Lanes: ' + '{:02.2f}'.format(curve_rad/1000) + 'Km'
    cv.putText(unwarped, rad_text, (30,70), font, 1.5, (255,255,255), 1, cv.LINE_AA)
    
    position_text = "Car's Position: " + pos_text + 'm'
    cv.putText(unwarped, position_text, (30,120), font, 1.5, (255,255,255), 1, cv.LINE_AA)
    
    return unwarped

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, name):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients for the most recent fit
        self.current_fit = []  
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # name of lines
        self.name = name

    # Gives the best fitting values of all the paramters
    def give_best_fit(self, fitx, fit_coeff, lane_inds, binarynonzero):

        nonzero = binarynonzero.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        
        # If polyfit coeff was found and we have a best fit then ignore if it is not an outlier
        if fit_coeff is None:
            self.detected = False
            print('no coeff detected')
        else:    
             # If polyfit coeff is detected and if there is a best fit we compare our current coeff with it to check if there is no major change
            if self.best_fit is not None:    
                self.diffs = abs(fit_coeff - self.best_fit)
                
                #If there is no major change or fluctuation in the current coeff we append it into a list for averaging and use the polynomial search we call this the perfecct fit
                if (self.diffs[0] < 0.001 and self.diffs[1] < 1 and self.diffs[2] < 160):
                    self.detected = True
                    self.current_fit.append(fit_coeff)
                    self.allx = nonzerox[lane_inds]
                    self.ally = nonzeroy[lane_inds]
                    print("perfect fit", self.diffs, fit_coeff)
                    
                # We again check this fit and if exceeds a certain threshold such that it becomes a outlier we use the best fit value instead of the current coeff and we use the slidewindow method   
                if (self.diffs[0] < 0.001 or self.diffs[1] < 0.5 or self.diffs[2] < 120):
                    
                    if (self.diffs[0] > 0.001 or self.diffs[1]>1 or self.diffs[2]>180) and len(self.current_fit)>0:
                        
                        
                        self.current_fit.append(self.current_fit[-1])
                        #self.detected = False
                        print("bad poly used previous value", self.diffs, 'previous val-', self.current_fit[-1])
                        
                        
                # we average to find the best fitting polynomial coeff    
                if len(self.current_fit)>2:
                    self.current_fit = self.current_fit[len(self.current_fit)-2:]
                    self.best_fit = np.average(self.current_fit, axis=0)
                    
            # In case no best fit is detected we make the current fit as the best fit            
            else:
                self.best_fit = fit_coeff
                print('im in no best fit')
                self.allx = nonzerox[lane_inds]
                self.ally = nonzeroy[lane_inds]
                
# 7. A pipeline for the Videos

# Finding the distortion coeff and camera matrix to use for undistorting images
mtx, dist = calibrateCam()

# Defining objects of line class
left_lane = Line('Left')
right_lane = Line('Right')

# Initializing object of 'VideoCapture' class to play videos
vid = cv.VideoCapture('harder_challenge_video.mp4')
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))

# For saving videos
out = cv.VideoWriter('harder_challenge_video_output.avi', cv.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

while vid.isOpened():

    # Reading video frames in a loop     
    ret, frame = vid.read()
    
    # If a frame is returned 
    if ret==True:
        # Undistort the frame and apply color and gradient threshold to isolate the lane lines
        combined_binary, undist = colorGradientThreshold(frame, mtx, dist)

        # Using perpective transform to give a top angle view at the road                  
        warped_binary, Minv = perspectiveTransform(combined_binary)
        
        # If even on of the lanes are not detected we use the sliding window method
        if not left_lane.detected or not right_lane.detected:
            out_img, left_fitx, right_fitx, ploty, left_fit, right_fit, left_lane_inds, right_lane_inds = fit_polynomial_window(warped_binary)
            cv.imshow('slide', out_img)

        # If both the lanes are detected we use the polynoial search method
        else:
            result, left_fitx, right_fitx, ploty, left_fit, right_fit, left_lane_inds, right_lane_inds = search_around_poly(warped_binary, left_fit, right_fit)
            res = cv.resize(result, (360,240))
            cv.imshow('Polynomial search', res)

        # We use the method of the line class to find the best fit line parameters
        left_lane.give_best_fit(left_fitx, left_fit, left_lane_inds, warped_binary)
        right_lane.give_best_fit(right_fitx, right_fit, right_lane_inds, warped_binary)
        
        # If both lanes have a best fit value then only the perspective will be wrapped back onto the original image
        if left_lane.best_fit is not None and right_lane.best_fit is not None:
            
            # Converting from top angle view to original view with annotated road area 
            unwarped = warpBack(warped_binary, left_lane.best_fit, right_lane.best_fit, ploty)

            # Calculating radius of curvature of lane
            curve_rad = measure_curvature_pixels(warped_binary, left_lane.allx, left_lane.ally, right_lane.allx, right_lane.ally)

            # Calculating postion of car wrt to the center of the lane
            pos_text = carPosition(unwarped, left_lane.best_fit, right_lane.best_fit)

            # Displaying calculated value onto the final frame
            final = display(unwarped, curve_rad, pos_text)
            
            cv.imshow('final', final)
            out.write(final)

        cv.imshow('Original', frame)
        cv.imshow('Combined Binary', combined_binary)
        cv.imshow('Warped Binary', warped_binary)
        
        key = cv.waitKey(1)

        if key == 32:
            cv.waitKey()
        elif key == ord('q'):
            break
    else:
        break
            
vid.release()
cv.destroyAllWindows()
