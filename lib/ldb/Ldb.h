/*
	LDB.h
	Created on: Apr 10, 2013
	Author: xinyang

    LDB - Local Difference Binary 
    Reference implementation of
    [1] Xin Yang and Kwang-Ting(Tim) Cheng. LDB: An Ultra-Fast Feature for 
	Scalable Augmened Reality on Mobile Device. In Proceedings of
    the IEEE International Symposium on Mixed and Augmented Reality(ISMAR2012).

    Copyright (C) 2012  The Learning-Based Multimedia, University of California, Santa Barbara
    Xin Yang, Kwang-Ting(Tim) Cheng.

    This file is part of LDB.

    LDB is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    LDB is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with LDB.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef LDB_H_
#define LDB_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

class LDB
{
public:

	int kBytes;
	
	LDB(int _bytes = 32, int _nlevels = 3, int _patchSize = 60);
	~LDB();

    // returns the descriptor size in bytes
    int descriptorSize() const;
    // returns the descriptor type
    int descriptorType() const;

    // Compute the LDB features and descriptors on an image
	void compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;
	
	// Compute the LDB for a resized image by employing a keypoint in the center of the image.
	void computeResized( const Mat& _image, Mat& _descriptors) const;
	// Compute the LDB for a resized image by employing several keypoints distributed on a grid.
	void computeResizedMultipoint( const Mat& _image, Mat& _descriptor, int _gridx, int _gridy) const;

protected:
	
	int nfeatures;
	double scaleFactor;
	int nlevels;
	int firstLevel;
	int patchSize;

};

typedef LDB LdbDescriptorExtractor;

#endif
