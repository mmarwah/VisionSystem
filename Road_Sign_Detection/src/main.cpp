/* Description: This file contains the code for CS7008 (VISION SYSTEM)
 * module assignment #1. 
 *
 * Task: #1: To identify RED boundaries of road signs in the provided image.
 *       #2: To identify black and white pixels inside the RED boundary of road sign.
 *
 * Author: Manas Marawaha (MSc. Mobile and Ubiquitous Computing)
 *         marawahm@tcd.ie
 *
 * Platform: Opencv V3.0 over C++ on LINUX Platform
*/

#include<iostream>
#include <iomanip>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

MatND gHistogram;

/* Description: Utility function to:
 *                  1). calculate and normalize the histogram of given image.
 *                  2). Calculate back projection of given image and training histogram.
 * Param 1: Image
 * Param 2: Number of bins
 * Param 3: Choice of function to perform.
 * Return: For Histogram Calc. and normalization function updates the gHistogram global 
 *         var with provided image histogram.
 *         For Back projection calculation funcion returns the Mat type image.
 * */
Mat utility( Mat image, int number_bins, int choice ) {
    Mat result;
    MatND temphist;
    int nChannels = image.channels();
    int* nChannels_ptr = new int[nChannels];
    int* nBins_ptr = new int[nChannels];
    float nChannel_range[] = {0.0, 255.0};
    const float* channel_range[] = {nChannel_range, nChannel_range, nChannel_range};

    for(int i=0; i<nChannels; i++) {
        nChannels_ptr[i] = i;
        nBins_ptr[i] = number_bins;
    }
    if ( choice == 1 ) {
        /* Histogram Calculation */
        calcHist(&image, 1, nChannels_ptr, Mat(), gHistogram, nChannels, nBins_ptr, channel_range);
        /* Histogram Normalization */
        normalize(gHistogram, gHistogram, 1.0);
    } else if ( choice == 2 ) {
        /* Back Projection Calclation */
        calcBackProject(&image, 1, nChannels_ptr, gHistogram, result, channel_range, 255.0); 
        return result;
    }
}

/* Description: Performance Metric Calculation:
 * Param 1: Pointer to processed image 
 * Param 2: Pointer to ground truth image 
*/
void performance_cal(Mat* processed_image, Mat* ground_truth) {
    CV_Assert( processed_image->type() == CV_8UC1 );
    CV_Assert( ground_truth->type() == CV_8UC1 );
    int FP = 0, FN = 0, TN = 0, TP = 0;
    double precision, recall, accuracy;

    for (int row=0; row < ground_truth->rows; row++)
        for (int col=0; col < ground_truth->cols; col++) {
            uchar image_val = processed_image->at<uchar>(row,col);
            uchar ground_val = ground_truth->at<uchar>(row,col);
            if (ground_val > 0)
                if (image_val > 0)
                    TP++;
                else FN++;
            else if (image_val > 0)
                FP++;
            else TN++;
        }
    precision = ((double) TP) / ((double) (TP+FP));
    recall = ((double) TP) / ((double) (TP+FN));
    accuracy = ((double) (TP+TN)) / ((double) (TP+TN+FP+FN));

    cout<<"TRUE POSITIVE:"<<TP<<endl<<"TRUE NEGATIVE:"<<TN<<endl;
    cout<<"FALSE POSITIVE:"<<FP<<endl<<"FALSE NEGATIVE:"<<FN<<endl;
    cout<<"PRECISION:"<<precision<< endl;
    cout<<"RECALL:"<< recall<< endl;
    cout<<"ACCURACY:"<< accuracy<< endl;
}

int main(int argc, const char** argv)
{
    /*--------------- START OF ASSINGMENT #1 PART 1: RED PIXEL DETECTION -----------------------*/
    Mat road_sign_image, training_image, ground_truth_image;
    Mat temp_image, ground_truth_binary, ground_truth_gray;
    Mat opened_image, dilated_image;
    Mat five_by_five_element(5,5,CV_8U,Scalar(1));
    
    /* Read Original, Training and Ground Truth image */
    road_sign_image = imread("./RoadSignsComposite1.JPG", -1);
    training_image = imread("./training.jpg", -1);
    ground_truth_image = imread("./RoadSignsCompositeGroundTruth.png", -1);

    if(!road_sign_image.data || !training_image.data || !ground_truth_image.data) {
        cout<<"Could not open or find the image\n";
        return 1;
    }

    /* Display original road sign image */
    namedWindow( "ORIGINAL ROAD SIGNS", CV_WINDOW_AUTOSIZE );
    imshow("ORIGINAL ROAD SIGNS", road_sign_image);
    waitKey(0);

    /* Change Color space of training images from BGR to HLS */
    cvtColor(training_image, temp_image, CV_BGR2HLS);
    /* Histogram Calculation and normalization */
    utility(temp_image, 7, 1);

    /* Change Color space of test images from BGR to HLS */
    cvtColor(road_sign_image, temp_image, CV_BGR2HLS);
    
    /* Back projetion calculation */
    Mat projected_image = utility(temp_image, 7, 2);
    
    /* Threshold operation on back projection of red boundary */
    Mat projected_image_binary;
    threshold(projected_image,projected_image_binary,128, 255, THRESH_BINARY | THRESH_OTSU);
    dilate(projected_image_binary,dilated_image,Mat());

    
    /* Bitwise AND of back projected image and original image 
     * to produce Red boundaries on black background */
    Mat projected_image_final;
    bitwise_and(road_sign_image, road_sign_image, projected_image_final, dilated_image);
    namedWindow( "RED PIXEL DETECTION", CV_WINDOW_AUTOSIZE );
    imshow("RED PIXEL DETECTION", projected_image_final);
    waitKey(0);

    /* Performance calculation */
    cout <<"PERFORMANCE CALCULATION FOR RED BOUNDARY DETECTION"<<endl;
    /* Convert Ground truth to grayscale */
    cvtColor(ground_truth_image, temp_image, CV_BGR2GRAY);
    Mat ground_truth_intermediate1, ground_truth_intermediate2;
    /* Thresholding grayscale to create region of interest 
     * (White boundaries on black background representing RED region) */
    threshold(temp_image, ground_truth_intermediate1, 80, 255, THRESH_BINARY_INV);
    threshold(temp_image, ground_truth_intermediate2, 5, 255, THRESH_BINARY_INV);
    bitwise_xor(ground_truth_intermediate1, ground_truth_intermediate2, ground_truth_binary);
    performance_cal(&projected_image_binary, &ground_truth_binary);

    /*--------------- END OF ASSINGMENT #1 PART 1: RED PIXEL DETECTION -----------------------*/

    /*----------- START OF ASSINGMENT #1 PART 2: BLACK AND WHITE PIXEL CLASSIFICATION ---------*/
    Mat drawing, resultant, resultant_binary;
    Mat resultant_gray;

    /* Applying dialation to improve connected boundary 
     * in order to find contours properly */
    dilate(projected_image_binary,dilated_image,Mat());

    /* Finding Connected component and holes using contour finding technique */
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(dilated_image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    drawing = Mat::zeros( projected_image_binary.size(), CV_8UC3 );

    /* Below given logic will only draw the leaf nodes 
     * (Child contours) found from the back projected image */
    for( int i = 0; i< contours.size(); i++ )
    {
        if(hierarchy[i][2]<0){

            drawContours( drawing, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy, 2 );
        }
    }

    /* Applying OPENING operation on contour image 
     * to reduce the noise in the background */
    morphologyEx(drawing, opened_image, MORPH_OPEN,five_by_five_element, Point(-1,-1), 2);

    /* Bitwise AND of contoured image and original image 
     * to produce ROI on black background */
    bitwise_and(opened_image, road_sign_image, resultant);

    /* Thresholding the Black and White ROI to produce Black 
     * and white pixel differentiation */
    cvtColor(resultant, resultant_gray, CV_BGR2GRAY);
    threshold( resultant_gray, resultant_binary, 85, 255, THRESH_BINARY );
    namedWindow( "BLACK & WHITE ROI", CV_WINDOW_AUTOSIZE );
    imshow( "BLACK & WHITE ROI", resultant_binary);
    waitKey(0);

    /* Performance calculation */
    cout <<"PERFORMANCE CALCULATION FOR BLACK AND WHITE ROI"<<endl;
    threshold(temp_image, ground_truth_binary, 120, 255, THRESH_BINARY| THRESH_OTSU);
    performance_cal(&resultant_binary, &ground_truth_binary);

    /*----------- END OF ASSINGMENT #1 PART 2: BLACK AND WHITE PIXEL CLASSIFICATION ---------*/

    return 0;
}
