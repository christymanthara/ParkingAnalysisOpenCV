#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;


Mat grayImage, bluredImage, detected_edges, final;
int lowThreshold = 80, upperThreshold;

static Mat CannyThreshold(Mat img)
{

//step1: grayscale
cvtColor(img,grayImage, COLOR_BGR2GRAY);

//step2: we apply gaussian blur
GaussianBlur( grayImage, bluredImage, Size( 3,3), 0, 0 );

//step3:Apply Canny edge detection
// Canny( bluredImage, detected_edges, lowThreshold, lowThreshold*3, kernel_size );

Canny( bluredImage, detected_edges, lowThreshold, lowThreshold*3);

// final = Scalar::all(0);
// colorImage.copyTo( final, detected_edges);
// imshow( window_name, final );
// imshow( window_name, detected_edges );

return detected_edges;
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

        //create a black image of the same size as the original image
        Mat blackimg = Mat::zeros(image.size(),CV_8UC3);

        imshow("Blackimage" + to_string(i),blackimg);
        //calling Canny
        final = CannyThreshold(image);
        imshow("Cannyimage" + to_string(i),final);
        

        i++;
        waitKey(0); 
    }
return (0);

}