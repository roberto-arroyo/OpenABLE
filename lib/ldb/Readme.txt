	libLDB
	Created on: May 27, 2014
	Author: xinyang, chonghuang

    LDB - Local Difference Binary 
    Reference implementation of
    [1] Yang, X., and K.-T. Cheng. Local Difference Binary for 
	Ultra-fast and Distinctive Feature Description. IEEE Trans. 
	on PAMI, vol. 36, no. 1, 2014.
    [2] Yang, X., and K.-T. Cheng. Learning Optimized Local 
	Difference Binariesfor Scalable Augmented Reality on Mobile 
	Devices. IEEE Trans. on VCG, 2014.
    [3] Yang, X., and K.-T. Cheng. LDB: An ultra-fast feature 
	for scalable Augmented Reality on mobile devices. 
	In Proc. of ISMAR, 2012.

    Copyright (C) 2012  The Learning-Based Multimedia, University of California, 
	Santa Barbara, Xin Yang, Kwang-Ting(Tim) Cheng.

    libLDB is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    libLDB is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with libLDB.  If not, see <http://www.gnu.org/licenses/>.


    Runing the Unit Tests
    
    This project has been tested with GCC 4.7.1, and OpenCV is also 
	required to build the binary. Make sure that the version of 
	OpenCV is 2.4.0 or above.

    ldb image1 image2 flag
	
    where flag determine whether to rotate the image window as follows:
	0  don’t rotate image window
    1  rotate image window.

    
 
