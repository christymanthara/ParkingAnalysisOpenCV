#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
 
using namespace cv;
using namespace std;
// namespace fs = std::experimental::filesystem;
 
const char* params
    = "{ help h         |           | Print usage }"
      "{ input          | vtest.avi | Path to a video or a sequence of image }"
      "{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";
 
int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, params);
    parser.about( "This program shows how to use background subtraction methods provided by "
                  " OpenCV. You can process both videos and images.\n" );
    if (parser.has("help"))
    {
        //print help information
        parser.printMessage();
    }





 
    //create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
    if (parser.get<String>("algo") == "MOG2")
        pBackSub = createBackgroundSubtractorMOG2();
    else
        pBackSub = createBackgroundSubtractorKNN();
 
    VideoCapture capture( samples::findFile( parser.get<String>("input") ) );
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open: " << parser.get<String>("input") << endl;
        return 0;
    }



//___________________________________________________________________________________


// //_____________________________________________________________________________________

 
    Mat frame, fgMask;
    RNG rng(12345);

    while (true) {
        capture >> frame;
        if (frame.empty())
            break;
 
        //update the background model
        pBackSub->apply(frame, fgMask);
 
        //get the frame number and write it on the current frame
        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
                  cv::Scalar(255,255,255), -1);
        stringstream ss;
        ss << capture.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
 
        //show the current frame and the fg masks
        imshow("Frame", frame);
        imshow("FG Mask", fgMask);


        //lets remove the shadows using thresholding

        threshold(fgMask, fgMask, 180, 255, cv::THRESH_BINARY);
        imshow("No Shadows", fgMask);

        // Applying the Opening Morphollogical operation to remove the points (Erosion+Dilation)
        morphologyEx( fgMask, fgMask, MORPH_OPEN, cv::getStructuringElement(MORPH_CROSS,Size(3,3) )); 
        imshow("Points Removed", fgMask);



        //now we find the contours
        

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours( fgMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
     
        //displaying the contours
        Mat drawing = Mat::zeros( fgMask.size(), CV_8UC3 );

        for( size_t i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( 0,0,255 );
            drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
        }

        imshow( "Contours", drawing );

        //filtering out the contours

        double min_carea = 200;
        // double carea = contourArea(contours);

        vector<vector<Point> > lcontours;
        for( size_t i = 0; i< contours.size(); i++ )
        {
            double area = contourArea(contours[i]);
            if (area > min_carea) {  // You can adjust this threshold
                Scalar color = Scalar(0, 0, 255);
                drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);


                Rect boundingBox = boundingRect(contours[i]);
                rectangle(drawing, boundingBox.tl(), boundingBox.br(), Scalar(0, 255, 0), 2);
                //-----------------------------------------------------------------------------------------------------------------

                //adding the bounding boxes

                
                String areaText = to_string(area); //display the area of the contours you obtained finally
                putText(drawing, areaText, contours[i][0], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }
        }

        imshow("Area filtered contours", drawing);


      



 
        //get the input from the keyboard
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
 
    return 0;
}