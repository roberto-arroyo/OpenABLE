/*****************************************************************************/

/**
 * @file    OpenABLE.cpp
 * @brief   Core functions of the OpenABLE toolbox
 * @author  Roberto Arroyo
 * @date    June 1, 2016
 */

/*****************************************************************************/

#ifndef OPENABLE_H
#define OPENABLE_H

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
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/objdetect.hpp>

// Other Includes
#include "../lib/ldb/Ldb.h"
#include "../lib/d_ldb/D_Ldb.h"

// Namespaces
using namespace std;
using namespace cv;

/*****************************************************************************/

// OpenABLE class
class OpenABLE{

    private:

        // Configuration: datasets parameters
        // If the dataset is a sequence images -> 0. If it is a video -> 1. Two videos -> 2
        int frame_mode;
        // Path where the images of the tested dataset are located (if Frame_mode = 0)
        string images_path;
        // Path where the left stereo images of the tested dataset are located (if Frame_mode = 0 and Camera_type = 1)
        string images_path_left;
        // Path where the right stereo images of the tested dataset are located (if Frame_mode = 0 and Camera_type = 1)
        string images_path_right;
	// Path where the calibration file for stereo images is located (if Frame_mode = 0 and Camera_type = 1)
        string calibration_path;
        // If the dataset is composed of images, specify the format (if Frame_mode = 0)
        string images_format;
        // Path where the video is located, including the video name (if Frame_mode = 1)
        string video_path_1;
        // Additional video path for processing two videos (if Frame_mode = 2)
        string video_path_2;
        // Numerical identifier for the first dataset image
        int initial_image;
        // Numerical identifier for the final dataset image
        int final_image;
 
        // Configuration: representation parameters
        // Name of the file where the similarity matrix is saved
        string txt_name;
        // Name of the image where the similarity matrix is printed
        string png_name;
        // If printed, you can choose the similarity matrix color
        int matrix_color;
        // If 1, the program shows the processed images
        int show_images;         

        // Configuration: description and matching parameters
        // Choose between monocular (0), stereo (1) or panoramic images (2)
        int camera_type;
        // Choose between global (0) or grid-based (1) description
        int description_type;
        // Number of bits used in the patch for global image description (if Description_type = 0)
        int patch_size;
        // Size of the grid applied in x (if Description_type = 1)
        int grid_x;
        // Size of the grid applied in y (if Description_type = 1)
        int grid_y;
        // Number of panoramas applied (if Camera_type = 2)
        int panoramas;
        // Choose between illumination invariance (1) or not (0)
        int illumination_invariance;
        // Alpha value (if Illumination_invariance = 1)
        float alpha;
        // Name of the descriptor applied in the image description
        string image_descriptor;
        // The number of images to be matched in sequence in each iteration
        int image_sequences;
        // Threshold value for considering loop closures (between 0 and 1) (disable = -1)
        float threshold;

        // Class attributes
        // Similarity matrix obtained after processing a complete sequence
        Mat similarity_matrix;
        // List of descriptors gradually obtained
	vector<Mat> descriptors;

	// Time attributes
        // Total time for describing all the images.
        double t_description;
        // Total time for matching all the images.
        double t_matching;
        // Average time for describing an image.
        double t_avg_description;
        // Average time for matching two images.
        double t_avg_matching;

    public:

        // Constructor
        OpenABLE(char* config_name);

        // Methods
        Mat global_description(Mat image_left, Mat image_right);
        Mat grid_description(Mat image_left, Mat image_right);
        Mat illumination_conversion(Mat image);
        int hamming_matching(Mat desc1, Mat desc2);
        float l2norm_matching(Mat desc1, Mat desc2);
        void similarity_matrix_normalization();
        void similarity_matrix_to_txt();
        void similarity_matrix_to_png();
        void compute_OpenABLE_monocular();
        void compute_OpenABLE_stereo();
        void compute_OpenABLE_panoramic();
        void compute_OpenABLE();
        void show_times();

};

#endif // OPENABLE_H

/*****************************************************************************/
