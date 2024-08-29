#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cmath>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;


Mat grayImage, bluredImage, detected_edges, final;
int lowThreshold = 100, upperThreshold;
std::vector<cv::Point> polygon_corners;

static Mat CannyThreshold(Mat img)
{

//step1: grayscale
cvtColor(img,grayImage, COLOR_BGR2GRAY);

//step2: we apply gaussian blur
GaussianBlur( grayImage, bluredImage, Size( 3,3), 0, 0 );

//step3:Apply Canny edge detection
// Canny( bluredImage, detected_edges, lowThreshold, lowThreshold*3, kernel_size );

Canny( bluredImage, detected_edges, lowThreshold, lowThreshold*3);

return detected_edges;
}


static Mat masking(Mat image,int i)
{
    Mat masked_image;

    //create a black image of the same size as the original image
        Mat blackimg = Mat::zeros(image.size(),CV_8UC1);
        // imshow("Blackimage" + to_string(i),blackimg);

    //taking the polygon area (hard coding)
        polygon_corners.push_back(cv::Point(243, 46));  
        polygon_corners.push_back(cv::Point(614, 718)); 
        polygon_corners.push_back(cv::Point(933, 585)); 
        polygon_corners.push_back(cv::Point(412, 19));  
        

        fillPoly(blackimg, polygon_corners, Scalar(255, 255, 255)); //fill the mask with white
        // imshow("Blackimage filled" + to_string(i),blackimg);

        //applyinng the mask to the original image
        bitwise_and(image,image,masked_image,blackimg);
        imshow("And Mask Applied" + to_string(i),masked_image);
        // bitwise_not(image,masked_image,blackimg);
        // imshow("Not Mask Applied" + to_string(i),masked_image); //using the not mask also removes the white lines into black

        return masked_image;
        
}

static Mat findlines(Mat img,int i)
{
    Mat finallines,grayImage;
    
    //step1: convert to grayscale 
    cvtColor(img, grayImage, COLOR_BGR2GRAY);
    imshow("Grayimage" + to_string(i),grayImage);
    // Step 2: applying threshhold
    threshold(grayImage, grayImage,130 , 255, cv::THRESH_BINARY); 
    // Step 3: bluring
    medianBlur(grayImage,grayImage,5);
    imshow("gray blur image",grayImage);


    grayImage.copyTo(finallines);
    return finallines;

}


int main() {
    string directory = "ParkingLot_dataset/sequence0/frames"; 

    int i =1;    
    for (const auto& entry : fs::directory_iterator(directory)) {
        
        string filepath = entry.path().string();
        string filename = entry.path().filename().string(); // getting the filename

        //load the file as image
        Mat image = cv::imread(filepath);

        if (image.empty()) {
            cout << "Could not open the images: " << filepath << std::endl;
            continue;
        }

        // cv::imshow(filename, image);
        
        imshow("image" + to_string(i),image);

        Mat masked = masking(image,i);

        
        
        

        // morphologyEx( final, final, MORPH_CLOSE, cv::getStructuringElement(MORPH_RECT,Size(5,5) )); 

        //detecting edges using the morphological operators
        morphologyEx( masked, final,MORPH_GRADIENT, Mat()); //works better
        threshold(final,final,100, 200, cv::THRESH_BINARY); 
        
        imshow("Closedimage" + to_string(i),final); 

        //now we use erosion and dilation to make the output more prominent     
        dilate(final,final,Mat());
        imshow("Dilatedimage" + to_string(i),final); 

        erode(final,final,Mat());
        imshow("Erodedimage" + to_string(i),final);  

        //uncomment for canny
        //calling Canny
        // final = CannyThreshold(final);
        // imshow("Cannyimage" + to_string(i),final);

        Canny( final, detected_edges, lowThreshold, 250);
        imshow("Cannyimage" + to_string(i),detected_edges); //calling canny directly

        // Probabilistic Hough Line Transform
        vector<Vec4f> lines; // will hold the results of the detection
        // HoughLinesP(final, lines, 10, CV_PI/180, 50, 100, 70 ); 

        // for(int i =0; i<(int)lines.size();i++)
        // {
        // //finding slope using using y2 - y1/x2-x1
        // double slope = (lines[i][3] - lines[i][1]) / (lines[i][2] - lines[i][0]);
        // double angle = atan(slope);
        // angle = angle * 180 / M_PI;
        
        // //finding arc tan of the slope gives us the angle

        // if(angle>=50 && angle<=70)
        // {
        //     std::cout<<angle;
        // cv::line(image,cv::Point(lines[i][0],lines[i][1]), cv::Point(lines[i][2],lines[i][3]),cv::Scalar(0, 0, 255),5,LINE_8,0);

        // }

        // }

        imshow("lined image",image);
        

        //trying with findlines()
        // findlines(masked,i);

        i++;
        waitKey(0); 
    }
return (0);

}