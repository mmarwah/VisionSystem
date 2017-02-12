/* Description: This file contains the header defines for CS7008 (VISION SYSTEM)
*  module assignment #1. 
**
** Task: #1: To identify RED boundaries of road signs in the provided
*            image.
**       #2: To identify black and white pixels inside the RED boundary
*            of road sign.
**
** Author: Manas Marawaha (MSc. Mobile and Ubiquitous Computing)
**         marawahm@tcd.ie
**
** Platform: Opencv V3.0 over C++ on LINUX Platform
**/

#ifndef __ROADSIGN_INCLUDED__
#define __ROADSIGN_INCLUDED__

#include<iostream>
#include <fstream>
#include<iomanip>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

Mat utility(Mat image, int number_bins, int choice);
void performance_cal(Mat* processed_image, Mat* ground_truth)

#endif
