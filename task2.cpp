#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>



using namespace std;
using namespace cv;

Mat colorImage;
Mat grayImage,bluredImage,detected_edges, final, croppedframe;
int alpha_slider, thresh1, angle=10;
double alpha;
double beta;
Mat img, imgHsv;

const int alpha_slider_max = 200;
const int angle_max = 359;
int lowThreshold = 0;
const int max_lowThreshold = 50;
// const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Canny Image";
Mat edges, rangeMask, rangeMask3C;

static Mat CannyThreshold(Mat grayImage)
{

//step2: we apply gaussian blur
GaussianBlur( grayImage, bluredImage, Size( 3,3), 0, 0 );

//step3:Apply Canny edge detection
// Canny( bluredImage, detected_edges, lowThreshold, lowThreshold*3, kernel_size );

Canny( bluredImage, detected_edges, lowThreshold, lowThreshold*3);

final = Scalar::all(0);
colorImage.copyTo( final, detected_edges);
imshow( window_name, final );

return final;

}


static void on_trackbar( int, void* )
{
GaussianBlur( grayImage, grayImage, Size( 3,3), 0, 0 );
Canny(grayImage, edges, 100, 200);


// Create a vector to store lines of the image

vector<Vec4i> lines;
// Apply Hough Transform
HoughLinesP(edges, lines, 1, CV_PI/180, thresh1, angle, 5);
// Draw lines on the image
for (size_t i=0; i<lines.size(); i++) {
    Vec4i l = lines[i];
    line(img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 3, LINE_AA);
}
// Show result image
imshow("HoughLines", img);
}


int main(int argc, char** argv)
{
img = cv::imread("input.jpg", IMREAD_COLOR);
imshow("Original image",img);

cvtColor(img, imgHsv, COLOR_BGR2HSV);
namedWindow("HSV Image", WINDOW_NORMAL);
imshow("HSV Image", imgHsv);

cvtColor(img,grayImage, COLOR_BGR2GRAY);

//---------------------------------------------

        Scalar lower_white(6, 25, 132);
        Scalar upper_white(161, 244, 255);

        // Create mask using the color range defined
        inRange(imgHsv, lower_white, upper_white, rangeMask);

        namedWindow("HSV ImageMask", WINDOW_NORMAL);
        imshow("HSV ImageMask", rangeMask);

        cvtColor(rangeMask, rangeMask3C, COLOR_GRAY2BGR);

        bitwise_and(img, rangeMask3C, final, rangeMask);
        namedWindow("Lines Masked", WINDOW_NORMAL);
        imshow("Lines Masked", final);

        //now i will apply closing to connect the lines

        morphologyEx( final, final, MORPH_CLOSE, cv::getStructuringElement(MORPH_RECT,Size(7,7) )); 
        imshow("Closed final", final);

//---------------------------------------------------------------





namedWindow("HoughLines", WINDOW_AUTOSIZE); 
char TrackbarName[50],TrackbarName2[50];
sprintf( TrackbarName, "Alpha x %d", alpha_slider_max );
createTrackbar( TrackbarName, "HoughLines", &thresh1, alpha_slider_max, on_trackbar );
sprintf( TrackbarName2, "Angle x %d", alpha_slider_max );
createTrackbar( TrackbarName2, "HoughLines", &angle, angle_max, on_trackbar );

waitKey(0);
return(0);

}

