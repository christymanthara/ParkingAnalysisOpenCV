



#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>



using namespace std;
using namespace cv;

Mat colorImage;
Mat grayImage,bluredImage,detected_edges, final, croppedframe;

int lowThreshold = 0;
const int max_lowThreshold = 100;
// const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Canny Image";

static void CannyThreshold(int, void*)
{

//step2: we apply gaussian blur
GaussianBlur( grayImage, bluredImage, Size( 5,5), 0, 0 );

//step3:Apply Canny edge detection
Canny( bluredImage, detected_edges, lowThreshold, lowThreshold*3, kernel_size );

final = Scalar::all(0);
colorImage.copyTo( final, detected_edges);
imshow( window_name, final );

}

int main(int argc, char** argv)
{
colorImage = cv::imread("input.jpg", IMREAD_COLOR);
imshow("Original image",colorImage);

//step1: convert to grayscale 
cvtColor(colorImage,grayImage, COLOR_BGR2GRAY);
namedWindow( window_name, WINDOW_AUTOSIZE );
// //creating the trackbar and calling it
// createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);


// i used threshold to segment out only the white lines using appropriate threshold values .
Mat thresh1;
threshold(grayImage, thresh1, 240, 255, cv::THRESH_BINARY);
imshow("thresh1",thresh1);





waitKey(0);
return(0);

}

