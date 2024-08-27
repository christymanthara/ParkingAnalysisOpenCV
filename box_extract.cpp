#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

int main() {
    std::string directory = "ParkingLot_dataset/sequence0/frames"; 

    int i =1;    
    for (const auto& entry : fs::directory_iterator(directory)) {
        
        std::string filepath = entry.path().string();
        std::string filename = entry.path().filename().string(); // getting the filename

        //load the file as image
        cv::Mat image = cv::imread(filepath);

        if (image.empty()) {
            cout << "Could not open the images: " << filepath << std::endl;
            continue;
        }

        // cv::imshow(filename, image);
        
        cv::imshow("image" + to_string(i),image);
        i++;
        cv::waitKey(0); 
    }
return (0);

}