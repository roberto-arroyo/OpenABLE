/*****************************************************************************/

/**
 * @file    D_Ldb.cpp
 * @brief   Core functions of the D-Ldb descriptor for stereo images
 * @author  Roberto Arroyo
 * @date    June 1, 2016
 */

/*****************************************************************************/

#ifndef D_LDB_H
#define D_LDB_H

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <limits>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>

// OpenCV Includes
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>

// Other Includes
#include "../ldb/Ldb.h"

// Namespaces
using namespace std;
using namespace cv;

/*****************************************************************************/

// D-Ldb class
class D_Ldb{

    private:

	// Calibration: stereo parameters
	// Camera focal length in pixels
	double Fx;
        // Fx = Fy if the stereo is rectified
	double Fy;
	// Horizontal pixelic central point
	double u0;
	// Vertical pixelic central point
	double v0;
        // Stereo camera baseline in milimeters
	double B;
        // Image width in pixels
	unsigned int width;
        // Image height in pixels
	unsigned int height;
	// Minimum disparity
	unsigned int min_disp;
	// Maximum disparity
	unsigned int max_disp;
        // 3x3 Left camera matrix
	Mat ML;
	// 3x3 Right camera Matrix
	Mat MR;
	// 3x3 Left rectification transform	
	Mat RL;
	// 3x3 Right rectification transform
	Mat RR;
	// 3x4 Left projection matrix rectified
	Mat PL;
	// 3x4 Right projection matrix rectified
	Mat PR;
	// 4x4 Disparity to depth mapping matrix
	Mat Q;
	// 3x3 Inverse left camera matrix
	Mat ML_inv;
	// 3x3 Inverse right camera matrix
	Mat MR_inv;
	// 3x3 Rotation matrix
	Mat R;
	// 3x1 Translation vector
	Mat T;
	// 5x1 Left camera distortion parameters
	Mat DL;
	// 5x1 Right camera distortion Parameters
	Mat DR;
	// Current undistorted and rectified left image
	Mat img_left;
	// Current undistorted and rectified right image
	Mat img_right;
	// Disparity map
	Mat disp;
	// Color disparity image
	Mat_<Vec3b> img_disp_c;
	// Mapping matrices necessary for image rectification and undistortion processes             
	Mat mxl, myl, mxr, myr;
	// Variable rgb_mode
	int colormap;
	// Logarithmic disparity representation  
	int use_log;

	// Some constant parameters for the stereo disparity computation
	int prefilter_size;
	int prefilter_cap;
	int sad_window_size;
	int min_disparity;
	int max_disparity;
	int uniqueness_ratio;
	int texture_threshold;
	int speckle_window_size;
	int speckle_range;

    public:

        // Constructor
        D_Ldb(char* calibration_name);

        // Methods
        void undistort_rectify_init(Size image_size);
	void undistort_rectify_images(Mat &img_left_d, Mat &img_right_d);
	double get_disparity(int u, int v);
	void depth_image_to_cv_image();
	Mat get_color_disparity_map();
        void compute_disparity(Mat &img_left_d, Mat &img_right_d);
        Mat compute_D_Ldb(Mat &img_left_d, Mat &img_right_d, vector<KeyPoint> &kpts, Mat &descriptor);

};

#endif // D_LDB_H

/*****************************************************************************/
