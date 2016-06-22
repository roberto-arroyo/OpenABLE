-------------------------------------------------------------------------------

OpenABLE
Copyright (c) 2016, Roberto Arroyo.
All Rights Reserved.

-------------------------------------------------------------------------------

1.INTRODUCTION

OpenABLE is an open toolbox that contributes different solutions and
functionalities to the research community in the topic of life-long visual
localization for autonomous vehicles.

The implemented code is in constant evolution, so improved versions will be
uploaded when needed. The management of the implementation is operated by
means of a repository created on GitHub, where the updates in the code are
done. Please, in case you find any bug, contact the main author:
roberto.arroyo@depeca.uah.es.

The code is opened to be used by the community attending to the copyright
described in "License.txt". In addition, if you use our open toolbox in any
publication or work, you must cite the corresponding papers where the theory
and the description of the implementation of OpenABLE has been published: 

* R. Arroyo, L. M. Bergasa and E. Romera, "OpenABLE: An Open-source Toolbox
for Application in Life-Long Visual Localization of Autonomous Vehicles",
submitted to Intelligent Transportation Systems Conference (ITSC), Rio de
Janeiro (Brazil), November 2016 (in review process).

* R. Arroyo, P. F. Alcantarilla, L. M. Bergasa and E. Romera, "Towards
Life-Long Visual Localization using an Efficient Matching of Binary Sequences
from Images", in IEEE International Conference on Robotics and Automation
(ICRA), pp. 6328-6335, Seattle, Washington (United States), May 2015.

* R. Arroyo, P. F. Alcantarilla, L. M. Bergasa, J. J. Yebes and S. Bronte,
"Fast and Effective Visual Place Recognition using Binary Codes and Disparity
Information", in IEEE/RSJ International Conference on Intelligent Robots and
Systems (IROS), pp. 3089-3094, Chicago, Illinois (United States), September
2014.

* R. Arroyo, P. F. Alcantarilla, L. M. Bergasa, J. J. Yebes and S. Gámez,
"Bidirectional Loop Closure Detection on Panoramas for Visual Navigation", in
IEEE Intelligent Vehicles Symposium (IV), pp. 1378-1383, Dearborn, Michigan
(United States), June 2014.

-------------------------------------------------------------------------------

2. HOW TO USE IT?

OpenABLE is designed in C++ using a Linux OS. OpenCV libraries (3.0 or higher)
are applied for computer vision functionalities and they must be installed to
employ our toolbox. Compilation is easy, you can use "gcc" and "cmake" for it,
a file called "CMakeLists.txt" is provided for this task.

The code files included in the project are conveniently explained and
commented. For a better comprehension, a test program is provided with the aim
of evaluating it, where a distance matrix is returned with the results obtained
in locations matching. A configuration file is also included for adjusting the
main parameters that can be modified by the user. You can apply different
configurations in your experiments. The execution of the test program can be
done using the following command after compilation:

./Test_OpenABLE Config.txt

You can modify and use the toolbox in your own ways. Our code can be applied
for different purposes related to localization, visual odometry or visual SLAM.
Adapt the code to employ it in your own interests!

NOTE: Although the code is designed in a Linux OS, it is easily adaptable for
application in Windows or MAC in a similar way.

-------------------------------------------------------------------------------

3. MAINTENANCE

OpenABLE is a project in constant evolution. Improved versions will be
conveniently uploaded to GitHub:

https://github.com/roberto-arroyo/OpenABLE

Check out the author's Web for information about updates:

http://www.robesafe.com/personal/roberto.arroyo/

-------------------------------------------------------------------------------
