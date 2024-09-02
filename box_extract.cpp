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


static Mat masking(Mat image,int i) //works perfectly
{
    Mat masked_image;

    //create a black image of the same size as the original image
        Mat blackimg = Mat::zeros(image.size(),CV_8UC1);
        // imshow("Blackimage" + to_string(i),blackimg);

    //taking the polygon area (hard coding)
        polygon_corners.push_back(cv::Point(243, 46));  
        polygon_corners.push_back(cv::Point(649, 705)); 
        polygon_corners.push_back(cv::Point(911, 596)); 
        polygon_corners.push_back(cv::Point(412, 19));  
        

        fillPoly(blackimg, polygon_corners, Scalar(255, 255, 255)); //fill the mask with white
        // imshow("Blackimage filled" + to_string(i),blackimg);

        //applyinng the mask to the original image
        bitwise_and(image,image,masked_image,blackimg);
        // imshow("And Mask Applied" + to_string(i),masked_image);
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

static Mat applyMSER(Mat img, int i)
{
        Mat output(img.size(),CV_8UC3);
        output = cv::Scalar(255,255,255);
        // cvtColor(img,img, COLOR_BGR2GRAY);
        vector<std::vector<cv::Point>> mpoints;
        std::vector< Rect > bboxes;
        //detect MSER features
        //using MSER
        Ptr<MSER> mser;
        mser = cv::MSER::create(5,800,8000,0.2,5,1000);//15,1000,8000,0.5,0.5,1000
        mser->detectRegions(img,mpoints,bboxes);

        //drawing the detected regions
        for (size_t i = 0; i < bboxes.size(); i++) {
        rectangle(img, bboxes[i], Scalar(0, 255, 0));  // Draw rectangle around the detected region
        }
        for (size_t i = 0; i < mpoints.size(); i++) {
        polylines(img, mpoints[i], true, Scalar(0, 255, 0), 2);
        }
        //drawing with random colors
        RNG rng;
        for(vector<vector<Point>>::iterator it=mpoints.begin();it!=mpoints.end();++it)
        {
            //generating random color
            cv::Vec3b c(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));

            //for each point in MSER set
            for (auto itPts = it->begin(); itPts != it->end(); ++itPts) {
            // If the point is part of the foreground (white in grayscale, for example)
            // we color it with the random color
            output.at<Vec3b>(*itPts) = c;
        }


        }




        // Display the image with detected regions
        imshow("MSER Regions"+ to_string(i), img);
        imshow("MSER Regions with random colors"+ to_string(i), output);
return img;

}


void sobelApplied(Mat img)
{
    Mat sobelx, sobely, sobelxy;
    Sobel(img, sobelx, CV_64F, 1, 0, 5);
    Sobel(img, sobely, CV_64F, 0, 1, 5);
    Sobel(img, sobelxy, CV_64F, 1, 1, 5);

    // Display Sobel edge detection images
    imshow("Sobel X", sobelx);
    
    imshow("Sobel Y", sobely);
 
    imshow("Sobel XY using Sobel() function", sobelxy);
    


}

void gammaCorrection(const cv::Mat& src, cv::Mat& dst, float gamma) {
    CV_Assert(gamma >= 0);
    cv::Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++) {
        lut.at<uchar>(i) = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    cv::LUT(src, lut, dst);
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

        Mat masked;

        masked = masking(image,i);

        //----------------------------------------------------------------gamma correction--------------------------------------
        float gamma = 3.2;
        cv::Mat gammaresult;
        gammaCorrection(masked, gammaresult, gamma);
        imshow("Gamma corrected image"+ to_string(i), gammaresult); 

        //----------------------------------------------------------------trying MSER---------------------------------------------
        // applyMSER(gammaresult,i);

        //----------------------------------------------------------------converting to gray-------------------------------------
        //make into gray
        Mat gray;
        cvtColor(gammaresult, gray, COLOR_BGR2GRAY);
        imshow("gray"+ to_string(i), gray);
        
        //-------------------------------------------------------applying morphological operation-----------------------------------
        morphologyEx( gray, final,MORPH_GRADIENT, Mat());
        imshow("Morph gradient image" + to_string(i),final);

        
        //-----------------------------------------------------------------applying thresholding------------------------------------
        Mat thresh;
        threshold(final, thresh, 180, 200, THRESH_BINARY + THRESH_OTSU); 
        imshow("Thresh lines"+ to_string(i), thresh);

        //-----------------------------------------------------------------Canny edge detector-----------------------------------------
        // Apply Canny edge detector
        Mat edges;
        Canny(thresh, edges, 80, 150, 3); 
        imshow("Canny Image"+ to_string(i), edges);

        //-----------------------------------------------------------------Hough transform---------------------------------------------
        vector<Vec4f> lines;
        HoughLinesP(edges, lines, 1, CV_PI / 180, 20, 10, 7);

        // Draw the detected lines on the original image
        for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        //------------------------------------finding angle--------------------------------------------
        float angle;
        angle = atan2(l[3] - l[1],l[2] - l[0])* 180.0 / CV_PI;
        cout<<"the angle"<<i<< "is"<<angle<<endl;
        //------------------------------------finding the length-----------------------------------------------
        double length = sqrt(pow(l[2] - l[0], 2) + pow(l[3] - l[1], 2));
        cout<<"the length of"<<i<< "is"<<length<<endl;

        line(image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);

        }

    // Display the result
    imshow("Detected White Lines", image);

        

        

        
        
        //sobelApplied(masked);

        Mat mserimg;

        gammaresult.copyTo(mserimg); 

        // mserimg = applyMSER(mserimg,i);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        //finding contours of mserimg
        // findContours(mserimg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        // //drawing the contours finally
        // for (size_t i = 0; i < contours.size(); i++) 
        // {
        // Scalar color = Scalar(0, 255, 0); // Green 
        // drawContours(mserimg, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
        // }

        // imshow("Contours in mserimg", mserimg);

        
        // image.copyTo(masked);

        // morphologyEx( final, final, MORPH_CLOSE, cv::getStructuringElement(MORPH_RECT,Size(5,5) )); 

        //detecting edges using the morphological operators

        cvtColor(masked,masked, COLOR_BGR2GRAY);
        // GaussianBlur( masked, masked, Size( 3,3), 0, 0 );
        morphologyEx( masked, final,MORPH_GRADIENT, Mat()); //works better
        threshold(final,final,100, 200, cv::THRESH_BINARY);  //adding thresh otsu
        
        // imshow("Closedimage" + to_string(i),final); 

        //now we use erosion and dilation to make the output more prominent     
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        // dilate(final,final,kernel);
        dilate(final,final,Mat());
        // imshow("Dilatedimage" + to_string(i),final); 

        erode(final,final,Mat());
        // imshow("Erodedimage" + to_string(i),final);  

        morphologyEx( final, final,MORPH_CLOSE, kernel); //closing
        // imshow("After Morph close" + to_string(i),final);


        //uncomment for canny
        //calling Canny
        // final = CannyThreshold(final);
        // imshow("Cannyimage" + to_string(i),final);

        Canny( final, detected_edges, lowThreshold, 250);
        // imshow("Cannyimage" + to_string(i),detected_edges); //calling canny directly

        

        
        // Probabilistic Hough Line Transform
        // vector<Vec4f> lines; // will hold the results of the detection
        HoughLinesP(detected_edges, lines, 1, CV_PI/180, 20, 10, 30 ); // use the default acumulator value rho =1
        Mat blackimg = Mat::zeros(image.size(),CV_8UC3);

        
        for(int i =0; i<(int)lines.size();i++)
        {
    
        cv::line(image,cv::Point(lines[i][0],lines[i][1]), cv::Point(lines[i][2],lines[i][3]),cv::Scalar(0, 0, 255),7,LINE_8,0);
  
        }

        // }
        // imshow("lined mask",blackimg);    
        // imshow("lined image",image);

        //finding contours
        findContours(detected_edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        

        //drawing the contours finally
        for (size_t i = 0; i < contours.size(); i++) 
        {
        Scalar color = Scalar(0, 255, 0); // Green 
        drawContours(blackimg, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
        }

        // imshow("Contours", blackimg);

        //trying with findlines()
        // findlines(masked,i);

        //--------------------------------------------------------------------------------------



        i++;
        waitKey(0); 
    }

return (0);

}