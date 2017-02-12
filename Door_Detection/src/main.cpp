/* Description: This file contains the code for CS7008 (VISION SYSTEM)
 * module assignment #2. 
 *
 * Task: #1: To identify moving door in the video.
 *
 * Author: Manas Marawaha (MSc. Mobile and Ubiquitous Computing)
 *         marawahm@tcd.ie
 *
 * Platform: Opencv V3.0 over C++ on LINUX Platform
*/

#include <iostream>
#include <iomanip>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"

#define PI 3.14159265358979323846
using namespace cv;
using namespace std;


/* Description: Draw hough lines on frame passed
 *             
 * Param 1: Current Frame
 * Param 2: Line segments vector
*/
void DrawLines(Mat result_image, vector<Vec4i> lines)
{
    for (vector<cv::Vec4i>::const_iterator current_line = lines.begin();
            (current_line != lines.end()); current_line++)
    {
        Point point1((*current_line)[0],(*current_line)[1]);
        Point point2((*current_line)[2],(*current_line)[3]);
        line( result_image, point1, point2, Scalar(0,0,255), 3, CV_AA );
    }
}

/* Description: Function to calcualte sobel edge detection. 
 *              
 * Param:  Input Image
 * Return: Gradient Image 
*/
Mat SobleEdgeDetection(Mat input)
{
    Mat input_gray, gradient;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    GaussianBlur(input, input, Size(3, 3), 0, 0, BORDER_DEFAULT);

    /* Convert input image into grayscale */
    cvtColor(input, input_gray, CV_BGR2GRAY);

    /* Generate grad_x and grad_y */
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /* Calculate Gradient X */
    Sobel(input_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    /* Calcualte Gradient Y */
    Sobel(input_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    /* Calcualte Total Gradient with approximation */
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradient);

    return gradient;
}

int main(int argc, const char** argv)
{

    VideoCapture vid;
    unsigned int frame_seq = 0;
    char overlay[25];
    Mat Intial_frame, current_frame, gmm_mask;
    Mat gmm_opened, gmm_closed, edge_image;
    Mat three_by_three_element(3,3,CV_8U,Scalar(1));

    /* Creating GMM Identifier */
    Ptr<BackgroundSubtractorMOG2> gmm = createBackgroundSubtractorMOG2();

    vid.open("./Door1.avi");
    if(!vid.isOpened())
        return -1;

    /* Counting Number of frames in the Video */
    if (vid.get(CV_CAP_PROP_FRAME_COUNT) < 1) {
        cout << "error: video file must have at least one frame\n";
        return(0);
    }

    /* Reding Initial frame */
    vid.read(Intial_frame);
    namedWindow("ORIGINAL", 1);

    /* Process video frame by frame */
    while(vid.isOpened())
    {
        frame_seq = vid.get(CV_CAP_PROP_POS_FRAMES);

        /* Check if next frame is not the last frame */
        if ( frame_seq + 1 <= vid.get(CV_CAP_PROP_FRAME_COUNT) ) {
            vid.read(current_frame);
        } else {
            cout << "END OF VIDEO\n";
            break;
        }

        /* Apply gaussian mixture model to detect motion */
        gmm->apply(current_frame, gmm_mask, 0.01);

        /* Applying CLOSING operation on GMM mask
         * to reduce the noise in the background */
        morphologyEx(gmm_mask, gmm_closed, MORPH_CLOSE,three_by_three_element, Point(-1,-1), 1);
        // morphologyEx(gmm_mask, gmm_closed, MORPH_CLOSE, Mat());

        /* Applying OPENING operation to reduce
         * the noise in the background */
        morphologyEx(gmm_closed, gmm_opened, MORPH_OPEN,three_by_three_element, Point(-1,-1), 1);
        //morphologyEx(gmm_closed, gmm_opened, MORPH_OPEN, Mat());

        /* Finding Connected component and holes using contour finding technique */
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(gmm_opened, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
        Mat gmm_object = Mat::zeros( gmm_closed.size(), CV_8UC3 );

        for( int i = 0; i< contours.size(); i++ )
            drawContours( gmm_object, contours, i, Scalar(255,255,255), CV_FILLED, 8, hierarchy, 2 );

        /* Edge detection using sobel */
        edge_image = SobleEdgeDetection(gmm_object);

        /* Generating Probabilistic hough line segments on original image */
        vector<Vec4i> hough_line_segments;
        HoughLinesP(edge_image, hough_line_segments, 1.0, PI/2, 200, 100, 100);
        DrawLines(current_frame, hough_line_segments);

        /* Print frame number on each frame */
        sprintf(overlay, "FRAME:%d", frame_seq);
        putText(current_frame, overlay, cvPoint(5,15), 
                FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,250), 1, CV_AA);

        /* Render processed video frame by frame */
        imshow("ORIGINAL", current_frame);

        if(waitKey(30) >= 0) break;
    }

    return 0;
}
