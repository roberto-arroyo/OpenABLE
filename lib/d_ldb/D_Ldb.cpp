/*****************************************************************************/

/**
 * @file    D_Ldb.cpp
 * @brief   Core functions of the D-Ldb descriptor for stereo images
 * @author  Roberto Arroyo
 * @date    June 1, 2016
 */

/*****************************************************************************/

#include "D_Ldb.h"

// Namespaces
using namespace std;
using namespace cv;

/*****************************************************************************/

/**
 * @brief This constructor loads the different calibration parameters
 * @param calibration_name Path where the calibration file is located
*/
D_Ldb::D_Ldb(char* calibration_name){

    //The calibration file is opened
    FileStorage fs;
    fs.open(calibration_name, FileStorage::READ);
    if (!fs.isOpened()){

        cout << "Couldn't open the calibration file. Check its location." << endl;

    }

    // Obtaining the calibration parameters

    fs["ML"] >> ML;
    fs["MR"] >> MR;
    fs["DL"] >> DL;
    fs["DR"] >> DR;
    fs["R"] >> R;
    fs["T"] >> T;
    fs["colormap"] >> colormap;
    fs["use_log"] >> use_log;

    Fx=ML.at<double>(0,0);
    Fy=ML.at<double>(1,1);
    u0=ML.at<double>(0,2);
    v0=ML.at<double>(1,2);
    B=-T.at<double>(0,0);

    fs.release();

    prefilter_size = 11;
    prefilter_cap = 31;
    sad_window_size = 11;
    min_disparity = 0;
    max_disparity = 128;
    uniqueness_ratio = 10;
    texture_threshold = 10;
    speckle_window_size = 0;
    speckle_range = 15;

    min_disp=min_disparity;
    max_disp=max_disparity;

}

/*****************************************************************************/

/**
 * @brief This function precomputes the LUTs for undistortion and rectification
 * @param image_size The size in pixels of the computed image
*/
void D_Ldb::undistort_rectify_init(Size image_size){

    width = image_size.width;
    height = image_size.height;
    stereoRectify(ML,DL,MR,DR,image_size,R,T,RL,RR,PL,PR,Q,CALIB_ZERO_DISPARITY,0,image_size);
    initUndistortRectifyMap(ML,DL,RL,PL,image_size,CV_16SC2,mxl,myl);
    initUndistortRectifyMap(MR,DR,RR,PR,image_size,CV_16SC2,mxr,myr);
    Fx=PR.at<double>(0,0);
    Fy=PR.at<double>(1,1);
    u0=PR.at<double>(0,2);
    v0=PR.at<double>(1,2);
    B=-PR.at<double>(0,3)/PR.at<double>(0,0);

}

/*****************************************************************************/

/**
 * @brief This function undistorts and rectifies the two images of the stereo
 * rig
 * @param img_left_d Left image
 * @param img_right_d Right image
*/
void D_Ldb::undistort_rectify_images(Mat &img_left_d, Mat &img_right_d){

    remap(img_left_d,img_left,mxl,myl,INTER_LINEAR);
    remap(img_right_d,img_right,mxr,myr,INTER_LINEAR);

}

/*****************************************************************************/

/**
 * @brief This function gets subpixelic disparity for an image position (u,v)
 * of the left image
 * @param u Image position in u
 * @param v Image position in v
 * @return The disparity in (u,v)
*/
double D_Ldb::get_disparity(int u, int v){

    double disp_32 = 0.0;

    if( u >= 0 && u < disp.cols && v >=0 && v < disp.rows ){

        if (disp.depth()==CV_32F)
            disp_32 = disp.at<float>(v,u);
        else // Disparity map is CV_16S depth
            disp_32 = disp.at<unsigned short>(v,u)/16.0;

    }

    return disp_32;

}

/*****************************************************************************/

/**
 * @brief This function converts the 16 or 32 bits depth image into a valid
 * color OpenCV image for visualization purposes
*/
void D_Ldb::depth_image_to_cv_image(){

    int i = 0, j = 0;
    double disp_32 = 0.0;
    unsigned char value = 0;

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(min_disparity,max_disparity-min_disparity,sad_window_size,0,0,0,prefilter_cap,uniqueness_ratio,speckle_window_size,speckle_range,StereoSGBM::MODE_HH);

    sgbm->compute(img_left,img_right,disp);
    img_disp_c.create(height,width);

    for( i = 0; i < img_disp_c.rows; i++){

        for( j = 0; j < img_disp_c.cols; j++){

            if (disp.depth()==CV_32F)
                disp_32 = disp.at<float>(i,j);
            else // Disparity map is CV_16S depth
                disp_32 = disp.at<unsigned short>(i,j)/16.0;

            if( disp_32 >= 0 ){

                if( use_log == 0 )
                    value = (unsigned char)(255.0*(disp_32-min_disp)/(max_disp-min_disp));
                else
                    value = (unsigned char)(255.0*log10((9.0/(max_disp-min_disp))*disp_32+min_disp-9.0*min_disp/(max_disp-min_disp)));

                // Grey Color Space -> 1 Channel Image
                if( img_disp_c.channels() == 1)
                    img_disp_c(i,j) = value;
                // RGB Gray Color Space -> 3 Channels Image
                else if( img_disp_c.channels() == 3 && colormap == 0)
                    img_disp_c(i,j)=Vec3b(value,value,value);
                // RGB Autumn Color Space
                else if( img_disp_c.channels() == 3 && colormap == 1)
                    img_disp_c(i,j)=Vec3b(0,value,255);
                // RGB Cool Color Space
                else if( img_disp_c.channels() == 3 && colormap == 2)
                    img_disp_c(i,j)=Vec3b(255,255-value,value);
                // RGB Copper Color Space
                else if( img_disp_c.channels() == 3 && colormap == 3)
                    img_disp_c(i,j)=Vec3b((unsigned char)0.4980*value,(unsigned char)0.7804*value,value);
                // RGB Hot Color Space
                else if( img_disp_c.channels() == 3 && colormap == 4){

                    //BGR Components
                    if( value < 92 )
                        img_disp_c(i,j)=Vec3b(0,0,(unsigned char)(2.6630*value+10));
                    else if( value >= 92 && value < 187)
                        img_disp_c(i,j)=Vec3b(0,(unsigned char)(2.68*value-246),255);
                    else
                        img_disp_c(i,j)=Vec3b((unsigned char)(3.75*value-701.25),255,255);

                }
                // RGB Spring Color Space
                else if( img_disp_c.channels() == 3 && colormap == 5)
                    img_disp_c(i,j)=Vec3b(255-value,value,255);
                // RGB Summer Color Space
                else if( img_disp_c.channels() == 3 && colormap == 6)
                    img_disp_c(i,j)=Vec3b(0,(unsigned char)(0.5*value+128),value);
                // RGB Winter Color Space
                else if( img_disp_c.channels() == 3 && colormap == 7)
                    img_disp_c(i,j)=Vec3b((unsigned char)(255-0.5*value),value,0);

            }

        }

    }

}

/*****************************************************************************/

/**
 * @brief This function returns the colored disparity map
 * @return The colored map
*/
Mat D_Ldb::get_color_disparity_map(){

    return img_disp_c.clone();

}


/*****************************************************************************/

/**
 * @brief This function calculates a D-LDB descriptor for the stereo pair
 * @param img_left_d Left image
 * @param img_right_d Right image
 *
*/
void D_Ldb::compute_disparity(Mat &img_left_d, Mat &img_right_d){

    // Process the stereo pair
    undistort_rectify_init(img_left_d.size());
    undistort_rectify_images(img_left_d,img_right_d);
    depth_image_to_cv_image();
    //imshow("Disparity", img_disp_c);
    //waitKey();

}

/*****************************************************************************/

/**
 * @brief This function calculates a D-LDB descriptor for the stereo pair
 * @param img_left_d Left image
 * @param img_right_d Right image
 * @param kpts Processed keypoints
 * @param descriptor Obtained descriptor
 *
*/
Mat D_Ldb::compute_D_Ldb(Mat &img_left_d, Mat &img_right_d, vector<KeyPoint> &kpts, Mat &descriptor){

    Mat descriptorDisp;
    Mat disparity;
    LDB ldb;

    // Process the stereo pair and D-LDB
    undistort_rectify_init(img_left_d.size());
    undistort_rectify_images(img_left_d,img_right_d);
    depth_image_to_cv_image();
    get_color_disparity_map().copyTo(disparity);
    cvtColor(disparity,disparity,COLOR_RGB2GRAY);
    //imshow("Disparity", disparity);
    //waitKey();
    ldb.compute(img_left_d,kpts,descriptor);
    ldb.compute(disparity,kpts,descriptorDisp);
    cv::hconcat(descriptor,descriptorDisp,descriptor);

    return descriptor;

}

/*****************************************************************************/
