/*****************************************************************************/

/**
 * @file    Test_OpenABLE.cpp
 * @brief   Program for testing the OpenABLE toolbox
 * @author  Roberto Arroyo
 * @date    June 1, 2016
 */

/*****************************************************************************/

// Includes
#include <iostream>
#include "src/OpenABLE.h"

// Namespaces
using namespace std;
using namespace cv;

/*****************************************************************************/

// Test program
/**
 * @brief This test program reads image or video sequences and detects loop
 * closures by applying the functionalities of the OpenABLE toolbox. Finally,
 * the program saves the similarity matrix scores into a file for subsequent
 * analysis.
*/
int main( int argc, char *argv[] ){

    if (argc==2){

        OpenABLE openABLE(argv[1]);
        openABLE.compute_OpenABLE();
        openABLE.show_times();
		
    }else{
	
        cout << "You must pass the path of the configuration file." << endl;
		
    }

}

/*****************************************************************************/
