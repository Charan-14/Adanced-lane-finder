# Advanced Lane finder
In this work using gradient and color thresholding we find the maximum concentration of pixels along the height of the image using a histogram sliding window approach and polynomial fitting.
Using perspective transform for better curve detection and also estimating the position of the car w.r.t the center of the lane.

### Pipeline
 
1. Computing the camera calibration matrix and distortion coefficients given a set of chessboard images to undistort raw images.
2. Using HLS colour transform and Sobelx gradient finding filter to create a thresholded binary image.
3. Applying perspective transform to have a birds eye view for lane finding.
4. Detecting lane pixels and fiting a line to find the lane boundary.
5. Warping the detected lane boundaries back onto the original image.
6. Determinig the curvature of the lane and vehicle position with respect to center.
7. Visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

![Screenshot from 2020-05-14 00-07-32](https://user-images.githubusercontent.com/58968984/81852301-72089280-9578-11ea-9b5c-3ce8084ffcb5.png)

1. Calibrating camera with chessboard images to find calibration matrix and distortion coefficients which are used to undistort raw images.

Since camera lenses due to their design cause bending of ray of light at the edges the image becomes distorted radially and tangentially. Thus we need to undistort these images and here we use chessboard images to achieve the same
We use a bunch of chessboard images clicked from various angles by the camera which is to be used. As we know the real world space 3d coordinates of the chessboard, we detect the chessboard corners using the openCV function cv2.findChessboardCorners(img, (rows,coloumns)) which returns the detected corners. To verify the detected corners we can use the cv2.drawChessboardCorners(img, (rows,coloumns), corners, ret) function. Now we have the image points as well we append all these points in a list and use it in this function cv2.calibrateCamera(obj_pts, img_pts, img.shape) which returns the camera matrix and distortion coefficients.

![Screenshot from 2020-04-30 02-39-44](https://user-images.githubusercontent.com/58968984/81864896-de8c8d00-958a-11ea-8f6f-4825cc5e88df.png)

Here the detected corners of chessboard image are shown

Finally we have the undistorted images which are the output of the function cv.undistort(frame, mtx, dist, None, mtx) which requires the camera matrix and the distortion coefficients.

![Screenshot from 2020-04-30 12-08-18](https://user-images.githubusercontent.com/58968984/81864743-ab49fe00-958a-11ea-9637-5670d3678a39.png)

![Screenshot from 2020-04-30 12-11-46](https://user-images.githubusercontent.com/58968984/81864791-bac94700-958a-11ea-88ab-fd51e461ad98.png)

![Screenshot from 2020-05-14 02-37-08](https://user-images.githubusercontent.com/58968984/81865701-d7b24a00-958b-11ea-8de7-a9655cda9965.png)

2. Using HLS & LAB Colour Space and Sobel Gradient Computation to isolate lane lines in all lighting conditions and create a binary thresholded image with no distortion.

Using HLS(Hue Lightness Saturation) color space we isolate the white lane line specifically using the lightness. Using the LAB (Lightness A*(green to red) B*(Blue to Yellow)) specifically the lightness and the B* colors we isolate the yellow lane line. The B* color channel isolates from to yellow colors.

![Screenshot from 2020-04-30 14-10-46](https://user-images.githubusercontent.com/58968984/81866709-36c48e80-958d-11ea-8f3b-a18c91d8b656.png)

![Screenshot from 2020-05-14 02-50-39](https://user-images.githubusercontent.com/58968984/81867478-67f18e80-958e-11ea-8cd2-17256db7a55e.png)

![Screenshot from 2020-05-14 02-50-19](https://user-images.githubusercontent.com/58968984/81867523-763faa80-958e-11ea-9830-1326fd3906b9.png)

Initially sobel gradient detection was used to find the edges in x direction using the sobel function in openCV. Gradient detection works good on areas with shadow but we skipped the idea of using as though it provided great results for the project video but in the challenge video many unwanted lines were detected. Thus, only using the color thresholds were used as they provided good results in both videos.

Sobel provided much better results than canny. Using sobel we can specifically find gradients in a particular direction here in the x direction such that lanes are visible.

![Screenshot from 2020-05-01 00-16-11](https://user-images.githubusercontent.com/58968984/81866802-5c519800-958d-11ea-957e-e48ae3f414c4.png)

Combination of both color and gradient threshold. The images were converted to binary for easier detection of activated pixels.

![Screenshot from 2020-05-01 01-21-03](https://user-images.githubusercontent.com/58968984/81903053-c34b6d00-95de-11ea-811e-ac9a8df5916f.png)

3. Perpective Transform for a birds eye view at the lanes to easily find the curve in the lanes.

We use the getPerspectiveTransform Function to return a transformation matrix which is used to warp the image from the source points to destination points. We select the source points such that the lanes appear almost parallel to each other in all conditions. The warpPerspective uses this matrix to transform the image to a new field of view.

![Screenshot from 2020-05-01 13-56-08](https://user-images.githubusercontent.com/58968984/81902814-66e84d80-95de-11ea-8768-0194e7b4b0d3.png)

4. Using Histogram to find maximum pixel concentration area to start line fitting.

By calculating the sum of pixels along the height of the image we plot a histogram which gives us the x coordinate where the sum is maximum. 

![Screenshot from 2020-05-01 17-01-46](https://user-images.githubusercontent.com/58968984/81906208-89309a00-95e3-11ea-99b7-98a7bab6e479.png)

we use this point as the starting point of creating a small window with a paricular margin and height within which we search for pixels. if we find more than 50 pixels we recenter the window at the mean of the pixels and continue this operation till the top of the image. All the pixels found within the sliding window are concatenated on top of each other and we have a resulting list of pixels of the lanes. These are used to fit a polynomial using np.polyfit function whch returns the 2nd order coefficients. These coefficients are finally used to plot a polynomial line using the equation x = ay^2 + by + c .

![Screenshot from 2020-05-02 01-41-02](https://user-images.githubusercontent.com/58968984/81906247-95b4f280-95e3-11ea-8bda-ec6b92731d99.png)

5. Using previous polynomial coefficients to find lane line around a certain margin instead of blind search

Since blind search is computationally expensive and unstable we use it as a starting point or in the case of some major curve change. Under normal curves searching around the margin of the polynomial is sufficient. Using the value of previous polynomial coeffcients we defing a margin of search of activated pixels and fit a polynomial corresponding to the result.
This is a much faster and stable way of searching.

![Screenshot from 2020-05-03 16-53-54](https://user-images.githubusercontent.com/58968984/81907040-ccd7d380-95e4-11ea-90ea-023091c7649b.png)

6. Warping the detected lane boundaries back onto the original image.

Using getPerspectiveTransform function we can also find the inverse matrix by switching up the source and destination points. This is then added with the original image to have our final result with visual display of lane boundaries.

![Screenshot from 2020-05-04 01-40-06](https://user-images.githubusercontent.com/58968984/81907710-ceee6200-95e5-11ea-9baf-fcb3ccc978fc.png)

![Screenshot from 2020-05-04 02-27-57](https://user-images.githubusercontent.com/58968984/81907747-e0376e80-95e5-11ea-8955-0e40cedc8958.png)

7. Calculating Radius of Curvature and Car's postion with respect to the centre of the lanes.

We first convert pixels to meters by estimating the length and width of the road an image covers which is 30m and 3.7m respectively. per pixel value is (actual length in meters/length in pixels). Radius of curvature is calculated by the formula ![Screenshot from 2020-04-29 10-52-31](https://user-images.githubusercontent.com/58968984/81908868-6bfdca80-95e7-11ea-8fac-85cccfb220cb.png)

As the camera is mounted at the centre of the car. So, the of car's position is the midpoint of the image. The centre of the lanes is the average of the x coordinate of left and right lane polynomial. By calculating their difference we get the car's deviation from the centre of the lanes.

![Screenshot from 2020-05-11 17-23-12](https://user-images.githubusercontent.com/58968984/81909381-27266380-95e8-11ea-86cd-438bfd9377d1.png)

## Discussion

### Parameter Tuning
The Line class ensures proper functioning and refinement of lane boundaries even when the binary image consists of blank image during shadows and during bright lighting and sunlight. It creates a best fit attribute for the lines and that is averaged for smoother results during outliers the previous best fit value is used. Using the previous best fir value when no lane pixels were found worked like a charm during underpass of the challenge video and the very bright areas of roads.

While this algorithm worked perfectly for both project video and the challenge video it failed at sharp curves in the harder challenge video 

The gradient detection method worked well for the project video but it provided too many unwanted lines in the challenge video so in the end only the color space threshold was used for both the project and challenge video.

Here are the video results [Project Video](https://github.com/Charan-14/Adanced-lane-finder/blob/master/output_videos/project_video_output.mp4)

The Challenge Video results [Challenge Video](https://github.com/Charan-14/Adanced-lane-finder/blob/master/output_videos/challenge_video_output.avi)

This video hit me in the face telling that your algorithm still has a lot to work on. It could not take sharp curves and the bery bright rays of the sun. The color and gradient threshold values both needed to be retuned, the perpective vertices need to be changed, and the polynomial outlier values needed to be changed. Also the margin for polynomial search needed to be reduced. All the parameters defining a good fit needed to be changed.

![Screenshot from 2020-05-17 03-41-57](https://user-images.githubusercontent.com/58968984/82152214-63262680-987d-11ea-8869-3fd68d291904.png)

The Harder Challenge Video results [Harder Challenge Video](https://github.com/Charan-14/Adanced-lane-finder/blob/master/output_videos/harder_challenge_video_output.mp4) 

### Improvements

The calculation of radius of curvature needs to be improved. I could use averaging of the lane pixel values

Right time when slidewindow should be used to aid polynomial search and the right time to use previous best fit value still needs to found for the harder challenge video.

The sharpest curve at the end of the harder challenge video is what broke the algorithm the only solution which comes to mind is using a better perspective transform.







