// utilities.hpp

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <opencv2/opencv.hpp> 

using namespace std;
using namespace cv;


Mat CannyThreshold(Mat img);


Mat masking(Mat image, int i);

Mat findlines(Mat img, int i);


Mat applyMSER(Mat img, int i);


void sobelApplied(Mat img);

void gammaCorrection(const cv::Mat& src, cv::Mat& dst, float gamma);

#endif // UTILITIES_HPP
