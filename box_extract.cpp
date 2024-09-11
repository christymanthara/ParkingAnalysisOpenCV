#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "utilities.hpp"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;


Mat grayImage, bluredImage, detected_edges, final;
int lowThreshold = 100, upperThreshold;
std::vector<cv::Point> polygon_corners;
Mat extImage;
RNG rnc;


//----------------------------------------------------------------------------------------------------------------------------------------------------
// Helper function to calculate the distance between two points
double pointDistance(Point p1, Point p2) {
    return sqrt((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y));
}

// Function to calculate the length of a line
double lineLength(Vec4i line) {
    Point p1(line[0], line[1]);
    Point p2(line[2], line[3]);
    return pointDistance(p1, p2);
}

// Function to find the distance between two lines (shortest distance between endpoints)
double lineDistance(Vec4i line1, Vec4i line2) {
    Point p1(line1[0], line1[1]); //starting point line 1
    Point p2(line1[2], line1[3]); //ending point line 1
    Point p3(line2[0], line2[1]); //starting point line 2
    Point p4(line2[2], line2[3]); //ending point line 2

    // Calculate all possible endpoint-to-endpoint distances
    double d1 = pointDistance(p1, p3);
    double d2 = pointDistance(p1, p4);
    double d3 = pointDistance(p2, p3);
    double d4 = pointDistance(p2, p4);

    // Return the minimum distance
    return min({d1, d3});
    // return min({d2, d4});
}

// Function to filter lines based on distance and keep the shorter one
void filterLines(vector<Vec4i>& lines, double distanceThreshold) {
    vector<Vec4i> filteredLines;
    vector<bool> keep(lines.size(), true); // Initially keep all lines

    for (size_t i = 0; i < lines.size(); i++) {
        if (!keep[i]) continue; // Skip if the line is already discarded

        for (size_t j = i + 1; j < lines.size(); j++) {
            if (!keep[j]) continue; // Skip if the line is already discarded

            // Calculate the distance between lines
            double distance = lineDistance(lines[i], lines[j]);

            if (distance < distanceThreshold) {
                // Compare the lengths and keep the shorter line
                double length1 = lineLength(lines[i]);
                double length2 = lineLength(lines[j]);

                if (length1 < length2) {
                    keep[j] = false; // Discard the longer line
                } else {
                    keep[i] = false; // Discard the longer line
                    break;
                }
            }
        }
    }

    // Collect the lines that are kept
    for (size_t i = 0; i < lines.size(); i++) {
        if (keep[i]) {
            filteredLines.push_back(lines[i]);
        }
    }

    lines = filteredLines; // Replace the original lines with the filtered lines
}


float findLineAngle(const Vec4f& line) {
    int dx = line[2] - line[0];
    int dy = line[3] - line[1];
    return (atan2(dy, dx) * 180.0 / CV_PI +180);

}

float findAngle(const Vec4f& line) {
    int dx = line[2] - line[0];
    int dy = line[3] - line[1];
    return (atan2(dy, dx) * 180.0 / CV_PI );

}

//------------------------------------------------------------checking close and collinearity-------------------------------------------------
bool checkCloseAndCollinear(const Vec4f& l1, const Vec4f& l2, float angleThreshold, double distanceThreshold) {
    float angle1 = findLineAngle(l1);
    float angle2 = findLineAngle(l2);
    
    if ((angle1 - angle2) > angleThreshold) {
        return false;  // Angles are too different
    }

    Point p1_start(l1[0], l1[1]); //x1,y1
    Point p1_end(l1[2], l1[3]); //x2,y2
    Point p2_start(l2[0], l2[1]); //x3,y3
    Point p2_end(l2[2], l2[3]); //x4,y4

    return (pointDistance(p1_end, p2_start) < distanceThreshold ||
            pointDistance(p1_end, p2_end) < distanceThreshold ||
            pointDistance(p1_start, p2_start) < distanceThreshold ||
            pointDistance(p1_start, p2_end) < distanceThreshold);
}

// Merge two lines into a single line by connecting the farthest points
Vec4f mergeLines(const Vec4i& l1, const Vec4i& l2) {
    Point p1_start(l1[0], l1[1]);
    Point p1_end(l1[2], l1[3]);
    Point p2_start(l2[0], l2[1]);
    Point p2_end(l2[2], l2[3]);

    Point start = p1_start;
    Point end = p1_end;
    
    // Find the farthest pair of points
    vector<Point> points = {p1_start, p1_end, p2_start, p2_end};
    double maxDist = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            double dist = pointDistance(points[i], points[j]);
            if (dist > maxDist) {
                maxDist = dist;
                start = points[i];
                end = points[j];
            }
        }
    }

    return Vec4i(start.x, start.y, end.x, end.y);
}

// Recursive function to merge lines that are close and collinear
void recursiveMerge(Vec4i& currentLine, vector<Vec4i>& lines, vector<bool>& merged, float angleThreshold, double distanceThreshold) {
    for (size_t i = 0; i < lines.size(); i++) {
        if (!merged[i]) {
            Vec4i nextLine = lines[i];

            if (checkCloseAndCollinear(currentLine, nextLine, angleThreshold, distanceThreshold)) {
                double llength = lineLength(currentLine);
                // if(llength<20)
                // {
                //     continue;
                // }
                // Merge the lines and mark the current line as merged
                currentLine = mergeLines(currentLine, nextLine);
                merged[i] = true;  // Mark the next line as merged

                // Recursively merge with more lines
                recursiveMerge(currentLine, lines, merged, angleThreshold, distanceThreshold);
            }
        }
    }
}

Point findmidpoint(Vec4i line)
{
    Point midpoint((line[0] + line[2]) / 2, (line[1] + line[3]) / 2);
    return midpoint;
}

Mat constructRectangles(Mat image, vector<Vec4i> lines, int distanceParallelThreshold) 
{
    vector<bool> isPaired(lines.size(), false); // Initialize to track if the line is making box1
    vector<bool> isPaired2(lines.size(), false); // Initialize to track which lines is making box2
    vector<bool> left_first(lines.size(), false); //initialize to track if the line is the first left line
    vector<int> count(lines.size());

    int smallestXIndex = -1;
    float smallestX = numeric_limits<float>::max();

    //Find the leftmost line
    for (size_t i = 0; i < lines.size(); i++) 
    {
        if (lines[i][0] < smallestX) 
        {
            smallestX = lines[i][0];
            smallestXIndex = i;
        }
    }

    if (smallestXIndex != -1)
    {
        left_first[smallestXIndex] = true; // Mark the smallest left index as true (this is the topmost line) //working checked
    }

   //nesting and computing
    for (size_t i = 0; i < lines.size(); i++) 
    {
        count[i] = 0; 
        Vec4f l1 = lines[i];  // Get the first line
        float angle1 = findAngle(l1);
        
        

        // Flip the line coordinates if the angle is negative
        if (angle1 < 0) {
            swap(l1[0], l1[2]);
            swap(l1[1], l1[3]);
        }

        if (left_first[i] == true) //checking if the left most line that is the first line
        { //for the first line we are giving both the boxing
            // count=1;
            isPaired[i]=true; 
            isPaired2[i]=true;
            count[i] = 2; 
            // Point p1(l1[0], l1[1]);  // Starting point of the first line
            // Point p2(l1[2], l1[3]);
            
        }


        //-------------------------------------------setting up the box variables---------------------------------------------
        if(isPaired[i]== true && isPaired2[i] ==true && left_first[i]!=true) //line has already been drawn into 2 boxes
        {
            continue;
        } 

        if (isPaired[i]==true){ //if one box is marked and the other is not marked,mark it
                        isPaired2[i]=true;
                        // break;
                        count[i]++; //count update
                    }
        else if(isPaired2[i]==true) //not needed i think
        {
            isPaired[i]=true;
            count[i]++; //count update
        }
        else{
            isPaired[i]=true; 
            count[i]++; //count update
        }
        //--------------------------------------------------------------------------------------------------------------------


        Vec4f closestLine;
        double closeDist = numeric_limits<double>::max();
        size_t closestLineIndex = -1;
        Point currentmidpoint = findmidpoint(l1);
        

         // To count how many boxes have been drawn


        for (size_t j = 0; j < lines.size(); j++) 
        {
            if (i == j || left_first[j]==true) continue;  // Skip if the line is the same or is the first line

            Vec4f l2 = lines[j];  // Get the second line
            float angle2 = findAngle(l2);

            // double closeDistance = numeric_limits<double>::max();

            // Flip the second line if its angle is negative
            if (angle2 < 0) {
                swap(l2[0], l2[2]);
                swap(l2[1], l2[3]);
                angle2 = -angle2;
            }

            if (fabs(angle1 - angle2) < 5) // Check if the two lines are nearly parallel (angle difference is small)
            {   

            //finding the midpoint of the second line
            Point closemidpoint = findmidpoint(l2);

            double mindist = pointDistance(currentmidpoint,closemidpoint);

            if (mindist < closeDist && isPaired[j]!=true && isPaired2[j]!=true) 
            {
                closeDist = mindist;
                closestLine = l2;
                closestLineIndex = j;
            }
            }
        }
            
            
                if (closestLineIndex != -1) 
                {
                Vec4f l2 = closestLine;


                // Check if the second line is directly under the first one
                Point midpoint1((l1[0] + l1[2]) / 2, (l1[1] + l1[3]) / 2);
                Point midpoint2((l2[0] + l2[2]) / 2, (l2[1] + l2[3]) / 2);
                double distance = pointDistance(midpoint1, midpoint2);


                // if (distance < distanceParallelThreshold) 
                // // if (distance < distanceParallelThreshold && midpoint2.y > midpoint1.y) 
                // {
                    // We found a pair of parallel lines that are close together and under each other
                    Point p1(l1[0], l1[1]);  // Starting point of the first line
                    Point p2(l1[2], l1[3]);  // Ending point of the first line
                    Point p3(l2[0], l2[1]);  // Starting point of the second line
                    Point p4(l2[2], l2[3]);  // Ending point of the second line

                    double deltaY_start = p3.y - p1.y;
                    double deltaX_start = p3.x - p1.x;
                    double angleleft = atan2(deltaY_start, deltaX_start) * 180 / CV_PI + 180;

                    double deltaY_end = p4.y - p2.y;
                    double deltaX_end = p4.x - p2.x;
                    double angleright = atan2(deltaY_end, deltaX_end) * 180 / CV_PI + 180;     


                 if (angleleft >= 50 && angleleft <= 60 && angleright >= 50 && angleright <= 60) 
            {
                    Point midpointl((l1[0] + l2[0]) / 2, (l1[1] + l2[1]) / 2);
                    Point midpointr((l1[2] + l2[2]) / 2, (l1[3] + l2[3]) / 2); 

                    // Draw the rectangle using these four points
                    line(image, p1, p3, Scalar(255, 255, 0), 2, LINE_AA); // Connect the start points
                    line(image, p2, p4, Scalar(255, 255, 0), 2, LINE_AA); // Connect the end points
                    line(image, p1, p2, Scalar(255, 255, 0), 2, LINE_AA); // Line along l1
                    line(image, p3, p4, Scalar(255, 255, 0), 2, LINE_AA); // Line along l2

                    //finding the angle formed by the new parallel lines connectng the lines up and below
                    putText(image, "Rectangle", midpoint1, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 200, 255), 1, LINE_AA);
                    putText(image, format("The angle is %.2f", angleleft), midpointl, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 200, 255), 1, LINE_AA);
                    putText(image, format("The angle is %.2f", angleright), midpointr, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 200, 255), 1, LINE_AA);

                    // if(angleleft > && angleleft < && angleright > && angleright <  )


                    count[closestLineIndex]++;
                    if (isPaired[i]==true){ //if one box is marked and the other is not marked then mark secoond line of the box
                        isPaired2[i]=true; //nesting
                        if(isPaired[closestLineIndex]==true)
                            isPaired2[closestLineIndex]=true;
                        else if(isPaired[closestLineIndex]!=true)
                            isPaired[closestLineIndex]=true;
                            
                        //break;
                    }
                    // else if(isPaired[i]==true){
                    //     isPaired[i]=true;
                        
            }
                    

                    

                    // If a pair is found, break and start the next iteration
                    // break;
                }
            }
        
    

// cout<<"now here";
    imshow("rectangles drawn",image);
    return image;
    
}


//--------------------------------------------------------------------------------------------------------------------------------------------------------

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
        
        // imshow("image" + to_string(i),image);

        Mat masked;

        masked = masking(image,i);
        image.copyTo(extImage);

        Mat randomcolored;
        image.copyTo(randomcolored);

        //------------------------------------------------------------trying with XYZ---------------------------------------
        Mat imgxyz;
        cvtColor(image, imgxyz, COLOR_BGR2XYZ);
        imshow("colorxyz"+ to_string(i), imgxyz);
        
        // ------------------------------------------------------applying masking to the image------------------------------
        masked = masking(imgxyz,i);
        //
        float gamma = 3.0; //3.2
        cv::Mat gammaresult;
        gammaCorrection(masked, gammaresult, gamma);
        imshow("Gamma corrected "+ to_string(i), gammaresult); 




        //----------------------------------------------------------------gamma correction--------------------------------------
        // float gamma = 3.0; //3.2
        // cv::Mat gammaresult;
        // gammaCorrection(masked, gammaresult, gamma);
        // imshow("Gamma corrected image"+ to_string(i), gammaresult); 

        //----------------------------------------------------------------trying MSER---------------------------------------------
        // applyMSER(gammaresult,i);

        //----------------------------------------------------------------converting to gray-------------------------------------
        //make into gray
        Mat gray;
        cvtColor(gammaresult, gray, COLOR_BGR2GRAY);
        imshow("gray"+ to_string(i), gray);
        
        //-------------------------------------------------------applying morphological operation-----------------------------------
        morphologyEx( gray, final,MORPH_GRADIENT, Mat());
        // imshow("Morph gradient image" + to_string(i),final);

        
        //-----------------------------------------------------------------applying thresholding------------------------------------
        Mat thresh;
        threshold(final, thresh, 180, 200, THRESH_BINARY + THRESH_OTSU); 
        // imshow("Thresh lines"+ to_string(i), thresh);

        //-----------------------------------------------------------------Canny edge detector-----------------------------------------
        // Apply Canny edge detector
        Mat edges;
        Canny(thresh, edges, 80, 150, 5, true); 
        // imshow("Canny Image"+ to_string(i), edges);


        //-----------------------------------------------------------------Finding Contours--------------------------------------------
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        //---------------------------------------------------------------drawing the contours-----------------------------------------
        Mat contourImg;
        image.copyTo(contourImg);
        for (size_t i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(0, 255, 0); // Green color for contours

        drawContours(contourImg, contours, (int)i, color, 2, LINE_8, hierarchy, 0); //uncomment to draw
        
    }
    imshow("Contour Image"+ to_string(i), contourImg);

        //-----------------------------------------------------------------Hough transform---------------------------------------------
        vector<Vec4i> lines;
        HoughLinesP(edges, lines, 1, CV_PI / 180, 20, 10, 7);


   


        //---------------------------------------------------------------Choosing lines which are smaller-------------------------------------
        double distanceThreshold = 2.0; // adjust this threshold= best =2
        filterLines(lines, distanceThreshold);





//{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}
        //------------------------------------------------------to check merging--------------------------------
        vector<Vec4i> mergedLines;  // for storing the merged lines
        vector<bool> merged(lines.size(), false);
        //---------------------------------------------------------------------------------------------------------                    
        // Draw the detected lines on the original image
        for (size_t i = 0; i < lines.size(); i++) 
        {
            
        Vec4f l = lines[i];
        //------------------------------------finding angle--------------------------------------------
        float angle;
        int dx=l[3] - l[1];
        int dy=l[2] - l[0];
        angle = atan2(dy,dx)* 180.0 / CV_PI;
        // cout<<"the angle"<<i<< "is"<<angle<<endl;
        //------------------------------------finding the length-----------------------------------------------
        double length = sqrt(pow(dy, 2) + pow(dx, 2));
        // cout<<"the length of"<<i<< "is"<<length<<endl;

        //----------    --------------------------plotting the midpoint and writing the point-------------------------------------------------
        Point midpoint((l[0] + l[2]) / 2, (l[1] + l[3]) / 2);

        double distanceThreshold = 15.0; //15
        double angleThreshold = 10.0;
        //=============================================Finding lines that are close to each other====================================================
        Vec4i l1 = lines[i];
        Point midpoint1((l1[0] + l1[2]) / 2, (l1[1] + l1[3]) / 2); //midpoint of line1

        // double llength = lineLength(l1);
        

        //applying recursive merging

        if (!merged[i]) {
            Vec4i currentLine = lines[i];  // Start with an unmerged line

            // Recursively merge lines that are close and collinear
            recursiveMerge(currentLine, lines, merged, angleThreshold, distanceThreshold);

            float angle;
            int dy=currentLine[3] - currentLine[1];
            int dx=currentLine[2] - currentLine[0];
            angle = atan2(dy,dx)* 180.0 / CV_PI;

            // After merging, add the combined line to the mergedLines vector
            mergedLines.push_back(currentLine);
        }



        }



        
        

        vector<Vec4i> filteredLines;
        //-------------------------------------------------------printing the recursively merged lines
        for (size_t i = 0; i < mergedLines.size(); i++) {
        Vec4i l = mergedLines[i];
        float angle;
        int dy=l[3] - l[1];
        int dx=l[2] - l[0];
        angle = atan2(dy,dx)* 180.0 / CV_PI;

        if (angle < 0) 
        {
        angle += 180.0; //making +ve and in a range
        }  

        Point midpoint = findmidpoint(l);
        double llength = lineLength(l);

        if (angle >= 3 && angle <= 20)
        {
        if(llength>=20)
        {
        line(randomcolored, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 2, LINE_AA);
        putText(randomcolored, format("The angle is %.2f", angle), midpoint, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 200, 255), 1, LINE_AA);
        filteredLines.push_back(l);
        }
        }


        //{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{alternate: userotatedrect}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

        



//==============================================================draw parallel lines into rectangles======================================================


        }
        imshow("Detected White Lines and Merged", randomcolored);
        int parallelthreshold = 200;
        Mat filteredRect = constructRectangles(randomcolored,filteredLines, parallelthreshold);
    

    // Display the result
    imshow("Detected White Lines", image);
    imshow("Extended White Lines", extImage);

    // imshow("Detected White Lines and Merged", randomcolored);
    // imshow("Filtered Rectangles", filteredRect);

        

        
        

        
        
        //sobelApplied(masked);

        Mat mserimg;

        gammaresult.copyTo(mserimg); 

   

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