/*****************************************************************************/

/**
 * @file    OpenABLE.cpp
 * @brief   Core functions of the OpenABLE toolbox
 * @author  Roberto Arroyo
 * @date    June 1, 2016
 */

/*****************************************************************************/

#include "OpenABLE.h"

// Namespaces
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

/*****************************************************************************/

/**
 * @brief This constructor loads the different configuration parameters
 * @param config_name Path where the configuration file is located
*/
OpenABLE::OpenABLE(char* config_name){

    // The configuration file of OpenABLE is opened
    FileStorage fs;
    fs.open(config_name, FileStorage::READ);
    if (!fs.isOpened()){

        cout << "Couldn't open the configuration file. Check its location." << endl;

    }

    // Obtaining the configuration parameters

    // Datasets parameters
    fs["Frame_mode"] >> frame_mode;
    fs["Images_path"] >> images_path;
    fs["Images_path_left"] >> images_path_left;
    fs["Images_path_right"] >> images_path_right;
    fs["Calibration_path"] >> calibration_path;
    fs["Images_format"] >> images_format;
    fs["Video_path_1"] >> video_path_1;
    fs["Video_path_2"] >> video_path_2;
    fs["Initial_image"] >> initial_image;
    fs["Final_image"] >> final_image;

    // Representation parameters
    fs["Txt_name"] >> txt_name;
    fs["Png_name"] >> png_name;
    fs["Matrix_color"] >> matrix_color;
    fs["Show_images"] >> show_images;

    // Description and matching parameters
    fs["Camera_type"] >> camera_type;
    fs["Description_type"] >> description_type;
    fs["Patch_size"] >> patch_size;
    fs["Grid_x"] >> grid_x;
    fs["Grid_y"] >> grid_y;
    fs["Panoramas"] >> panoramas;
    fs["Illumination_invariance"] >> illumination_invariance;
    fs["Alpha"] >> alpha;
    fs["Image_descriptor"] >> image_descriptor;
    fs["Image_sequences"] >> image_sequences;
    fs["Threshold"] >> threshold;

    fs.release();

}

/*****************************************************************************/

/**
 * @brief This function computes a global description for an image
 * @param image_left Computed image (left image in the stereo case)
 * @param image_right Right image in the stereo case
 * @return Obtained descriptor
*/
Mat OpenABLE::global_description(Mat image_left, Mat image_right){

    Mat image;
    if (illumination_invariance == 0)
        cvtColor(image_left, image, COLOR_BGR2GRAY);
    else
        illumination_conversion(image_left).copyTo(image);

    Mat descriptor;

    if (strcmp(image_descriptor.c_str(),"ORB")==0) patch_size=128;
    // Resize the image
    Mat image_resized;
    resize(image,image_resized,Size(patch_size,patch_size),0,0,INTER_LINEAR);
    Mat disparity_resized;

    // Select the central keypoint
    vector<KeyPoint> kpts;
    KeyPoint kpt;
    kpt.pt.x = patch_size/2+1;
    kpt.pt.y = patch_size/2+1;
    if ((strcmp(image_descriptor.c_str(),"SIFT")==0) ||
        (strcmp(image_descriptor.c_str(),"SURF")==0))
        kpt.size = 40.0;
    else
        kpt.size = 1.0;
    kpt.angle = 0.0;
    kpts.push_back(kpt);
    
    // Compute global descriptor
    if (strcmp(image_descriptor.c_str(),"BRIEF")==0){

        Ptr<DescriptorExtractor> brief = BriefDescriptorExtractor::create();
        brief->compute(image_resized,kpts,descriptor);

    }else if (strcmp(image_descriptor.c_str(),"BRISK")==0){

        Ptr<DescriptorExtractor> brisk = BRISK::create(5.85,8.2);
        brisk->compute(image_resized,kpts,descriptor);

    }else if (strcmp(image_descriptor.c_str(),"ORB")==0){

        Ptr<DescriptorExtractor> orb = ORB::create(500,1.2,8,30,0,2,ORB::HARRIS_SCORE,31);
        orb->compute(image_resized,kpts,descriptor);

    }else if (strcmp(image_descriptor.c_str(),"FREAK")==0){

        Ptr<DescriptorExtractor> freak = FREAK::create(true,true,22.0,4,vector<int>());
        freak->compute(image_resized,kpts,descriptor);

    }else if (strcmp(image_descriptor.c_str(),"LDB")==0){

        LDB ldb;
        ldb.compute(image_resized,kpts,descriptor);

    }else if (strcmp(image_descriptor.c_str(),"D-LDB")==0){

        if (camera_type == 1){

            Mat descriptorDisp;
            Mat disparity;
            LDB ldb;
            D_Ldb d_ldb(&calibration_path[0u]);
            d_ldb.compute_disparity(image_left,image_right);
            d_ldb.get_color_disparity_map().copyTo(disparity);
            cvtColor(disparity,disparity,COLOR_RGB2GRAY);
            resize(disparity,disparity_resized,Size(patch_size,patch_size),0,0,INTER_LINEAR);
            ldb.compute(image_resized,kpts,descriptor);
            ldb.compute(disparity_resized,kpts,descriptorDisp);
            cv::hconcat(descriptor,descriptorDisp,descriptor);

        } else
            cout << "D-LDB is only available in stereo mode." << endl;

    }else if (strcmp(image_descriptor.c_str(),"HOG")==0){

        vector<float> descriptors;
        vector<Point> locations;
        HOGDescriptor hog(Size(patch_size,patch_size), Size(16,16), Size(8,8), Size(8,8), 9, -1.0, 0.2, true, HOGDescriptor::DEFAULT_NLEVELS);
        hog.compute(image_resized, descriptors, Size(patch_size,patch_size), Size(0,0), locations);
        Mat descriptor_aux(descriptors,true);
        transpose(descriptor_aux,descriptor_aux);
        descriptor_aux.copyTo(descriptor);

    }else if (strcmp(image_descriptor.c_str(),"SIFT")==0){

        Ptr<DescriptorExtractor> sift = SIFT::create();
        sift->compute(image_resized,kpts,descriptor);

    }else if (strcmp(image_descriptor.c_str(),"SURF")==0){

        Ptr<DescriptorExtractor> surf = SURF::create();
        surf->compute(image_resized,kpts,descriptor);

    }else{

        cout << "The selected descriptor is not available in OpenABLE." << endl;

    }

    return descriptor;

}

/*****************************************************************************/

/**
 * @brief This function computes a grid-based description for an image
 * @param image_left Computed image (left image in the stereo case)
 * @param image_right Right image in the stereo case
 * @return Obtained descriptor
*/
Mat OpenABLE::grid_description(Mat image_left, Mat image_right){

    Mat image;
    if (illumination_invariance == 0)
        cvtColor(image_left, image, COLOR_BGR2GRAY);
    else
        illumination_conversion(image_left).copyTo(image);

    Mat descriptor;

    // Select the grid keypoints
    vector<KeyPoint> kpts;
    int dx = image.cols/grid_x;
    int dy = image.rows/grid_y;

    for(int i = 0; i < grid_x; i++){

        for(int j = 0; j < grid_y; j++){

            KeyPoint kpt;
            kpt.pt.x = i*dx + dx/2;
            kpt.pt.y = j*dy + dy/2;
            if ((strcmp(image_descriptor.c_str(),"SIFT")==0) ||
                (strcmp(image_descriptor.c_str(),"SURF")==0))
                kpt.size = 40.0;
            else
                kpt.size = 1.0;
            kpts.push_back(kpt);

        }

    }

    // Compute grid descriptor
    if (strcmp(image_descriptor.c_str(),"BRIEF")==0){

        Ptr<DescriptorExtractor> brief = BriefDescriptorExtractor::create();
        brief->compute(image,kpts,descriptor);

    }else if (strcmp(image_descriptor.c_str(),"BRISK")==0){

        Ptr<DescriptorExtractor> brisk = BRISK::create(5.85,8.2);
        brisk->compute(image,kpts,descriptor);

    }else if (strcmp(image_descriptor.c_str(),"ORB")==0){

        Ptr<DescriptorExtractor> orb = ORB::create(500,1.2,8,30,0,2,ORB::HARRIS_SCORE,31);
        orb->compute(image,kpts,descriptor);

    }else if (strcmp(image_descriptor.c_str(),"FREAK")==0){

        Ptr<DescriptorExtractor> freak = FREAK::create(true,true,22.0,4,vector<int>());
        freak->compute(image,kpts,descriptor);

    }else if (strcmp(image_descriptor.c_str(),"LDB")==0){

        LDB ldb;
        ldb.compute(image,kpts,descriptor);

    }else if (strcmp(image_descriptor.c_str(),"D-LDB")==0){

        if (camera_type == 1){

            Mat descriptorDisp;
            Mat disparity;
            LDB ldb;
            D_Ldb d_ldb(&calibration_path[0u]);
            d_ldb.compute_disparity(image_left,image_right);
            d_ldb.get_color_disparity_map().copyTo(disparity);
            cvtColor(disparity,disparity,COLOR_RGB2GRAY);
            ldb.compute(image,kpts,descriptor);
            ldb.compute(disparity,kpts,descriptorDisp);
            cv::hconcat(descriptor,descriptorDisp,descriptor);

        } else
            cout << "D-LDB is only available in stereo mode." << endl;

    }else if (strcmp(image_descriptor.c_str(),"HOG")==0){

        vector<float> descriptors;
        vector<Point> locations;
        HOGDescriptor hog(Size(patch_size,patch_size), Size(16,16), Size(8,8), Size(8,8), 9, -1.0, 0.2, true, HOGDescriptor::DEFAULT_NLEVELS);
        hog.compute(image, descriptors, Size(patch_size,patch_size), Size(0,0), locations);
        Mat descriptor_aux(descriptors,true);
        transpose(descriptor_aux,descriptor_aux);
        descriptor_aux.copyTo(descriptor);

    }else if (strcmp(image_descriptor.c_str(),"SIFT")==0){

        Ptr<DescriptorExtractor> sift = SIFT::create();
        sift->compute(image,kpts,descriptor);

    }else if (strcmp(image_descriptor.c_str(),"SURF")==0){

        Ptr<DescriptorExtractor> surf = SURF::create();
        surf->compute(image,kpts,descriptor);

    }else{

        cout << "The selected descriptor is not available in OpenABLE." << endl;

    }

    //Stack the descriptors obtained per each grid in one row
    Mat grid_descriptor = Mat::zeros(1,descriptor.cols*descriptor.rows,CV_8UC1);
    int k = 0;

    for( int j = 0; j < descriptor.rows; j++)
    {
        for( int i = 0; i < descriptor.cols; i++ )
        {
            *(grid_descriptor.ptr<unsigned char>(0)+k)=*(descriptor.ptr<unsigned char>(j)+i);
            k++;
        }
    }

    return grid_descriptor;

}

/*****************************************************************************/

/**
 * @brief Converts an image to illumination invariant image.
 * @param image Computed image
 * @return Illumination invariant image
*/
Mat OpenABLE::illumination_conversion(Mat image){

    vector<Mat> channels(3);
    split(image, channels);

    Mat imageB,imageG,imageR;
    Mat imageI = Mat(Size(image.cols,image.rows),CV_32FC1);
    Mat imageI8U = Mat(Size(image.cols,image.rows),CV_8UC1);

    channels[0].convertTo(imageB, CV_32FC1, 1.0/255.0, 0);
    channels[1].convertTo(imageG, CV_32FC1, 1.0/255.0, 0);
    channels[2].convertTo(imageR, CV_32FC1, 1.0/255.0, 0);

    float valueG, valueB, valueR;

    for (int i = 0; i < imageI.rows; i++){

        for (int j = 0; j < imageI.cols; j++){
 
            if (imageG.at<float>(i,j) != 0)
                valueG=log(imageG.at<float>(i,j));
            else 
                valueG=0;

            if (imageB.at<float>(i,j) != 0)
                valueB=alpha*log(imageB.at<float>(i,j));
            else 
                valueB=0;

            if (imageR.at<float>(i,j) != 0)
                valueR=(1-alpha)*log(imageR.at<float>(i,j));
            else 
                valueR=0;

            imageI.at<float>(i,j) = 0.5 + valueG - valueB - valueR;

            if (imageI.at<float>(i,j)<0) imageI.at<float>(i,j) = 1;

        }

    }

    imageI.convertTo(imageI8U, CV_8UC1, 255.0, 0);

    return imageI8U;

}

/*****************************************************************************/

/**
 * @brief This method computes the Hamming distance between two binary
 * descriptors
 * @param desc1 First descriptor
 * @param desc2 Second descriptor
 * @return Hamming distance between the two descriptors
*/
int OpenABLE::hamming_matching(Mat desc1, Mat desc2){

    int distance = 0;

    if (desc1.rows != desc2.rows || desc1.cols != desc2.cols || desc1.rows != 1 || desc2.rows != 1){

        cout << "The dimension of the descriptors is different." << endl;
        return -1;

    }

    for (int i = 0; i < desc1.cols; i++){

        distance += (*(desc1.ptr<unsigned char>(0)+i))^(*(desc2.ptr<unsigned char>(0)+i));

    }

    return distance;

}

/*****************************************************************************/

/**
 * @brief This method computes the l2-norm matching between two vector-based
 * descriptors
 * @param desc1 First descriptor
 * @param desc2 Second descriptor
 * @return Distance between the two descriptors
*/
float OpenABLE::l2norm_matching(Mat desc1, Mat desc2){

    float distance = 0.0;

    if (desc1.rows != desc2.rows || desc1.cols != desc2.cols || desc1.rows != 1 || desc2.rows != 1){

        cout << "The dimension of the descriptors is different." << endl;
        return -1;

    }

    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(desc1,desc2,matches);

    for(unsigned int i = 0; i < matches.size(); i++){

        distance += matches[i].distance;

    }

    return distance;

}

/*****************************************************************************/

/**
 * @brief This method normalizes the similarity matrix
*/
void OpenABLE::similarity_matrix_normalization(){

    int nimgs = similarity_matrix.rows;
    double min, max;

    minMaxLoc(similarity_matrix, &min, &max);

    for (int i = 0; i < nimgs; i++){

        for(int j = 0; j < nimgs; j++){

            if (threshold == -1){

                similarity_matrix.at<float>(i,j) = similarity_matrix.at<float>(i,j)/max;

            }else{

                if ((similarity_matrix.at<float>(i,j)/max)<threshold){

                    similarity_matrix.at<float>(i,j) = 0;

                }else{

                    similarity_matrix.at<float>(i,j) = 1;

                }

            }

        }

    }

}

/*****************************************************************************/

/**
 * @brief This method saves the similarity matrix into a .txt file
*/
void OpenABLE::similarity_matrix_to_txt(){

    int nimgs = similarity_matrix.rows;

    ofstream outFile;
    outFile.open(txt_name.c_str(),ofstream::out | ofstream::trunc);

    if (!outFile){

       cout << "Couldn't open file to save the similarity matrix." << endl;

    }

    for (int i = 0; i < nimgs; i++){

        for(int j = 0; j < nimgs; j++){

            outFile << similarity_matrix.at<float>(i,j) << " ";

        }

        outFile << endl;

    }

    outFile.close();

}

/*****************************************************************************/

/**
 * @brief This method saves the similarity matrix into a .png file
*/
void OpenABLE::similarity_matrix_to_png(){

    int nimgs = similarity_matrix.rows;
    Mat similarity_color(nimgs,nimgs,CV_32FC3);
    Mat similarity_color_8U(nimgs,nimgs,CV_8UC3);
    float similarity;

    for (int i = 0; i < nimgs; i++){

        for(int j = 0; j < nimgs; j++){

            similarity = similarity_matrix.at<float>(i,j);

            if (matrix_color == 0){

                similarity_color.ptr<Point3_<float> >(i,j)->x = similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->y = similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->z = similarity;

            }else if (matrix_color == 1){

                similarity_color.ptr<Point3_<float> >(i,j)->x = 0;
                similarity_color.ptr<Point3_<float> >(i,j)->y = 0;
                similarity_color.ptr<Point3_<float> >(i,j)->z = similarity;

            }else if (matrix_color == 2){

                similarity_color.ptr<Point3_<float> >(i,j)->x = 0;
                similarity_color.ptr<Point3_<float> >(i,j)->y = similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->z = 0;

            }else if (matrix_color == 3){

                similarity_color.ptr<Point3_<float> >(i,j)->x = similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->y = 0;
                similarity_color.ptr<Point3_<float> >(i,j)->z = 0;

            }else if (matrix_color == 4){

                similarity_color.ptr<Point3_<float> >(i,j)->x = 1-similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->y = 1-similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->z = similarity;

            }else if (matrix_color == 5){

                similarity_color.ptr<Point3_<float> >(i,j)->x = 1-similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->y = similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->z = 1-similarity;

            }else if (matrix_color == 6){

                similarity_color.ptr<Point3_<float> >(i,j)->x = similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->y = 1-similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->z = 1-similarity;

            }else if (matrix_color == 7){

                similarity_color.ptr<Point3_<float> >(i,j)->x = 0;
                similarity_color.ptr<Point3_<float> >(i,j)->y = similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->z = 1;

            }else if (matrix_color == 8){

                similarity_color.ptr<Point3_<float> >(i,j)->x = 1;
                similarity_color.ptr<Point3_<float> >(i,j)->y = 1-similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->z = similarity;

            }else if (matrix_color == 9){

                similarity_color.ptr<Point3_<float> >(i,j)->x = similarity*0.4980;
                similarity_color.ptr<Point3_<float> >(i,j)->y = similarity*0.7804;
                similarity_color.ptr<Point3_<float> >(i,j)->z = similarity;

            }else if (matrix_color == 10){

                similarity_color.ptr<Point3_<float> >(i,j)->x = 1-similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->y = similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->z = 1;

            }else if (matrix_color == 11){

                similarity_color.ptr<Point3_<float> >(i,j)->x = 0;
                similarity_color.ptr<Point3_<float> >(i,j)->y = similarity*0.5+0.5;
                similarity_color.ptr<Point3_<float> >(i,j)->z = similarity;

            }else{

                similarity_color.ptr<Point3_<float> >(i,j)->x = similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->y = similarity;
                similarity_color.ptr<Point3_<float> >(i,j)->z = similarity;

            }

        }

    }

    similarity_color.convertTo(similarity_color_8U, CV_8UC1, 255, 0);
    imwrite(png_name,similarity_color_8U);

}

/*****************************************************************************/

/**
 * @brief This method computes OpenABLE for a complete monocular image dataset
*/
void OpenABLE::compute_OpenABLE_monocular(){

    Mat image;
    char image_path[500];
    VideoCapture video;
    double t1, t2;
    int i = 0;
    int num_images = final_image - initial_image;

    if (frame_mode == 2)
       num_images = num_images * 2;

    similarity_matrix = Mat::ones(num_images,num_images,CV_32FC1);

    if ((frame_mode == 1) || (frame_mode == 2)){
        
        video.open(video_path_1);
        video.set(CAP_PROP_POS_FRAMES,initial_image-1);

    }
    
    // This loop computes each iteration of OpenABLE
    while(i < num_images){

        // Read the current image
        if (frame_mode == 0){

            sprintf(image_path,"%s%06d.%s",images_path.c_str(),i+initial_image,images_format.c_str());
            cout << "Reading image: " << image_path << endl;
            image = imread(image_path,1);
            if (image.data == NULL){
                cout << "The dataset path is not valid or it does not exist." << endl;
                break;
            }

            if (show_images == 1){

                Mat showed_image;
                resize(image, showed_image,Size((int)(image.cols*0.5),(int)(image.rows*0.5)),INTER_LINEAR);
                imshow(images_path, showed_image); waitKey(25);

            }

        }else if ((frame_mode == 1) || (frame_mode == 2)){

            cout << "Reading video frame: " << i+initial_image << endl;
            video >> image;
            if (image.data == NULL){
                cout << "The dataset path is not valid or it does not exist." << endl;
                break;
            }

            if (show_images == 1){

                Mat showed_image;
                resize(image, showed_image,Size((int)(image.cols*0.5),(int)(image.rows*0.5)),INTER_LINEAR);
                imshow(video_path_1, showed_image); waitKey(CAP_PROP_FPS);

            }

        }else{

            cout << "Choose a correct Frame_mode (between 0 and 2)." << endl;
            break;

        }

        // Image description
        t1 = getTickCount();
        if (description_type == 0)
            descriptors.push_back(global_description(image,image));
        else if (description_type == 1)
            descriptors.push_back(grid_description(image,image));
        else{
            cout << "Choose a correct Description_type (between 0 and 1)." << endl;
            break;
        }
        t2 = getTickCount();
        t_description += 1000.0*(t2-t1)/getTickFrequency();

        // Image matching 
        for (int j = i; j >= 0; j--){

            float distance = 0.0;
            float mult = 1.0;
            int num_sequences;

            if (j < image_sequences){
                num_sequences = j;
                mult = ((float)(image_sequences))/((float)(num_sequences+1));
            }else
                num_sequences = image_sequences;

            for (int k = num_sequences; k >= 0 ; k--){

                if ((strcmp(image_descriptor.c_str(),"BRIEF")==0) ||
                    (strcmp(image_descriptor.c_str(),"BRISK")==0) ||
                    (strcmp(image_descriptor.c_str(),"ORB")==0)   ||
                    (strcmp(image_descriptor.c_str(),"FREAK")==0) ||
                    (strcmp(image_descriptor.c_str(),"LDB")==0)){

                    t1 = getTickCount();
                    distance = distance + hamming_matching(descriptors[i-k],descriptors[j-k]);

                }else if ((strcmp(image_descriptor.c_str(),"HOG")==0)  ||
                          (strcmp(image_descriptor.c_str(),"SIFT")==0) ||
                          (strcmp(image_descriptor.c_str(),"SURF")==0)){

                    t1 = getTickCount();
                    distance = distance + l2norm_matching(descriptors[i-k],descriptors[j-k]);
               
                }

            }
 
            similarity_matrix.at<float>(i,j)=distance*mult;
            similarity_matrix.at<float>(j,i)=similarity_matrix.at<float>(i,j);

            t2 = getTickCount();
            t_matching += 1000.0*(t2-t1)/getTickFrequency();

        }

        i++;

        if ((frame_mode == 2) && (i == (num_images/2))){

            video.release();
            video.open(video_path_2);
            video.set(CAP_PROP_POS_FRAMES,initial_image-1);

        }

    }

    // The final times are saved for studying performance.
    t_avg_description = t_description / ((float)(num_images));
    t_avg_matching = t_matching / (((float)(num_images)*(float)(num_images))/2);

    cout << "Writing the similarity matrix on disk: please wait..." << endl;
    similarity_matrix_normalization();
    similarity_matrix_to_txt();
    similarity_matrix_to_png();

}

/*****************************************************************************/

/**
 * @brief This method computes OpenABLE for a complete stereo image dataset
*/
void OpenABLE::compute_OpenABLE_stereo(){

    Mat image_left, image_right;
    char image_path_left[500], image_path_right[500];
    double t1, t2;
    int i = 0;
    int num_images = final_image - initial_image;
    similarity_matrix = Mat::ones(num_images,num_images,CV_32FC1);

    // This loop computes each iteration of OpenABLE
    while(i < num_images){

        // Read the current image
        if (frame_mode == 0){

            sprintf(image_path_left,"%s%06d.%s",images_path_left.c_str(),i+initial_image,images_format.c_str());
            cout << "Reading image: " << image_path_left << endl;
            image_left = imread(image_path_left,1);
            if (image_left.data == NULL){
                cout << "The dataset path for left images is not valid or it does not exist." << endl;
                break;
            }

            sprintf(image_path_right,"%s%06d.%s",images_path_right.c_str(),i+initial_image,images_format.c_str());
            cout << "Reading image: " << image_path_right << endl;
            image_right = imread(image_path_right,1);
            if (image_right.data == NULL){
                cout << "The dataset path for right images is not valid or it does not exist." << endl;
                break;
            }

            if (show_images == 1){

                Mat showed_image;
                resize(image_left, showed_image,Size((int)(image_left.cols*0.5),(int)(image_left.rows*0.5)),INTER_LINEAR);
                imshow(images_path, showed_image); waitKey(25);

            }

        }else{

            cout << "Stereo mode only allows sequences of stereo images (video is not available in this case)." << endl;
            break;

        }

        t1 = getTickCount();
        if (description_type == 0)
            descriptors.push_back(global_description(image_left,image_right));
        else if (description_type == 1)
            descriptors.push_back(grid_description(image_left,image_right));
        else{
            cout << "Choose a correct Description_type (between 0 and 1)." << endl;
            break;
        }
        t2 = getTickCount();
        t_description += 1000.0*(t2-t1)/getTickFrequency();

        // Image matching
        for (int j = i; j >= 0; j--){

            float distance = 0.0;
            float mult = 1.0;
            int num_sequences;

            if (j < image_sequences){
                num_sequences = j;
                mult = ((float)(image_sequences))/((float)(num_sequences+1));
            }else
                num_sequences = image_sequences;

            for (int k = num_sequences; k >= 0 ; k--){

                if ((strcmp(image_descriptor.c_str(),"BRIEF")==0) ||
                    (strcmp(image_descriptor.c_str(),"BRISK")==0) ||
                    (strcmp(image_descriptor.c_str(),"ORB")==0)   ||
                    (strcmp(image_descriptor.c_str(),"FREAK")==0) ||
                    (strcmp(image_descriptor.c_str(),"LDB")==0)   ||
                    (strcmp(image_descriptor.c_str(),"D-LDB")==0)){

                    t1 = getTickCount();
                    distance = distance + hamming_matching(descriptors[i-k],descriptors[j-k]);

                }else if ((strcmp(image_descriptor.c_str(),"HOG")==0)  ||
                          (strcmp(image_descriptor.c_str(),"SIFT")==0) ||
                          (strcmp(image_descriptor.c_str(),"SURF")==0)){

                    t1 = getTickCount();
                    distance = distance + l2norm_matching(descriptors[i-k],descriptors[j-k]);

                }

            }

            similarity_matrix.at<float>(i,j)=distance*mult;
            similarity_matrix.at<float>(j,i)=similarity_matrix.at<float>(i,j);

            t2 = getTickCount();
            t_matching += 1000.0*(t2-t1)/getTickFrequency();

        }

        i++;

    }

    // The final times are saved for studying performance.
    t_avg_description = t_description / ((float)(num_images));
    t_avg_matching = t_matching / (((float)(num_images)*(float)(num_images))/2);

    cout << "Writing the similarity matrix on disk: please wait..." << endl;
    similarity_matrix_normalization();
    similarity_matrix_to_txt();
    similarity_matrix_to_png();

}

/*****************************************************************************/

/**
 * @brief This method computes OpenABLE for a complete stereo image dataset
*/
void OpenABLE::compute_OpenABLE_panoramic(){

    Mat image;
    Mat subpanorama;
    char image_path[500];
    VideoCapture video;
    double t1, t2;
    int i = 0;
    int num_images = final_image - initial_image;

    if (frame_mode == 2)
       num_images = num_images * 2;

    similarity_matrix = Mat::ones(num_images,num_images,CV_32FC1);

    if ((frame_mode == 1) || (frame_mode == 2)){

        video.open(video_path_1);
        video.set(CAP_PROP_POS_FRAMES,initial_image-1);

    }

    // This loop computes each iteration of OpenABLE
    while(i < num_images){

        // Read the current image
        if (frame_mode == 0){

            sprintf(image_path,"%s%06d.%s",images_path.c_str(),i+initial_image,images_format.c_str());
            cout << "Reading image: " << image_path << endl;
            image = imread(image_path,1);
            if (image.data == NULL){
                cout << "The dataset path is not valid or it does not exist." << endl;
                break;
            }

            if (show_images == 1){

                Mat showed_image;
                resize(image, showed_image,Size((int)(image.cols*0.5),(int)(image.rows*0.5)),INTER_LINEAR);
                imshow(images_path, showed_image); waitKey(25);

            }

        }else if ((frame_mode == 1) || (frame_mode == 2)){

            cout << "Reading video frame: " << i+initial_image << endl;
            video >> image;
            if (image.data == NULL){
                cout << "The dataset path is not valid or it does not exist." << endl;
                break;
            }

            if (show_images == 1){

                Mat showed_image;
                resize(image, showed_image,Size((int)(image.cols*0.5),(int)(image.rows*0.5)),INTER_LINEAR);
                imshow(video_path_1, showed_image); waitKey(CAP_PROP_FPS);

            }

        }else{

            cout << "Choose a correct Frame_mode (between 0 and 2)." << endl;
            break;

        }

        // Image description
        t1 = getTickCount();
        for (int j=0; j<panoramas; j++){
            // A descriptor is processed for each subpanorama in this case
            image(Rect(image.cols/panoramas*j,0,image.cols/panoramas,image.rows)).copyTo(subpanorama);
            if (description_type == 0)
                descriptors.push_back(global_description(subpanorama,subpanorama));
            else if (description_type == 1)
                descriptors.push_back(grid_description(subpanorama,subpanorama));
            else{
                cout << "Choose a correct Description_type (between 0 and 1)." << endl;
                break;
            }
        }
        t2 = getTickCount();
        t_description += 1000.0*(t2-t1)/getTickFrequency();

        // Image matching using cross-correlation of subpanoramas
        for (int j = i; j >= 0; j--){

            float distance = 0.0;
            float mult = 1.0;
            int num_sequences;

            if (j < image_sequences){
                num_sequences = j;
                mult = ((float)(image_sequences))/((float)(num_sequences+1));
            }else
                num_sequences = image_sequences;

            for (int k = num_sequences; k >= 0 ; k--){

                if ((strcmp(image_descriptor.c_str(),"BRIEF")==0) ||
                    (strcmp(image_descriptor.c_str(),"BRISK")==0) ||
                    (strcmp(image_descriptor.c_str(),"ORB")==0)   ||
                    (strcmp(image_descriptor.c_str(),"FREAK")==0) ||
                    (strcmp(image_descriptor.c_str(),"LDB")==0)){

                    t1 = getTickCount();

                    float distanceMin = std::numeric_limits<float>::max();
                    float distanceSub = 0.0;

                    for (int m = 0; m < panoramas; m++){

                        for (int n = 0; n < panoramas; n++){

                            distanceSub = hamming_matching(descriptors[(panoramas*(i-k))+m],descriptors[(panoramas*(j-k))+n]);
                            if (distanceSub<distanceMin)
                                distanceMin=distanceSub;

                        }

                    }

                    distance = distance + distanceMin;

                }else if ((strcmp(image_descriptor.c_str(),"HOG")==0)  ||
                          (strcmp(image_descriptor.c_str(),"SIFT")==0) ||
                          (strcmp(image_descriptor.c_str(),"SURF")==0)){

                    t1 = getTickCount();

                    float distanceMin = std::numeric_limits<float>::max();
                    float distanceSub = 0.0;

                    for (int m = 0; m < panoramas; m++){

                        for (int n = 0; n < panoramas; n++){

                            distanceSub = l2norm_matching(descriptors[(panoramas*(i-k))+m],descriptors[(panoramas*(j-k))+n]);
                            if (distanceSub<distanceMin)
                                distanceMin=distanceSub;

                        }

                    }

                    distance = distance + distanceMin;

                }

            }

            similarity_matrix.at<float>(i,j)=distance*mult;
            similarity_matrix.at<float>(j,i)=similarity_matrix.at<float>(i,j);

            t2 = getTickCount();
            t_matching += 1000.0*(t2-t1)/getTickFrequency();

        }

        i++;

        if ((frame_mode == 2) && (i == (num_images/2))){

            video.release();
            video.open(video_path_2);
            video.set(CAP_PROP_POS_FRAMES,initial_image-1);

        }

    }

    // The final times are saved for studying performance.
    t_avg_description = t_description / ((float)(num_images));
    t_avg_matching = t_matching / (((float)(num_images)*(float)(num_images))/2);

    cout << "Writing the similarity matrix on disk: please wait..." << endl;
    similarity_matrix_normalization();
    similarity_matrix_to_txt();
    similarity_matrix_to_png();

}

/*****************************************************************************/

/**
 * @brief This method computes OpenABLE
*/
void OpenABLE::compute_OpenABLE(){

    if (camera_type == 0)
        compute_OpenABLE_monocular();
    else if (camera_type == 1)
        compute_OpenABLE_stereo();
    else if (camera_type == 2)
        compute_OpenABLE_panoramic();
    else
        cout << "Choose a correct Camera_type (between 0 and 2)." << endl;

}

/*****************************************************************************/

/**
 * @brief This method prints information about computational times
*/
void OpenABLE::show_times(){

    cout << "Computational times in ms" << endl;
    cout << "Total time for describing all the images: " << t_description << endl;
    cout << "Total time for matching all the images: " << t_matching << endl;
    cout << "Average time for describing an image: " << t_avg_description << endl;
    cout << "Average time for matching two images: " << t_avg_matching << endl;

}

/*****************************************************************************/
