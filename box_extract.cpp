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



//---------------------------------------------positive slopes checker--------------------------------------------------------------------------------
vector<Vec4i> checkPositiveSlope(const vector<Vec4i>& lines) {
    vector<Vec4i> positiveLines;

    for (const auto& line : lines) {
        // Extract points of the line
        Point pt1(line[0], line[1]);
        Point pt2(line[2], line[3]);

        // Calculate the slope as y2-y1 / x2-x1
        float slope = (pt2.y - pt1.y) / (float)(pt2.x - pt1.x);

        // Checking if slope is negative
        if (slope < 0) {
            // swap the points to ensure positive slope
            swap(pt1, pt2);
            // cout<<"pt1"<<pt1<<"and pt2"<<pt2<<endl;
        }

        // Store the line with positive slope
        positiveLines.push_back(Vec4i(pt1.x, pt1.y, pt2.x, pt2.y));
    }

    return positiveLines;
}

//-----------------------------------------------------------function to join the midpoints of the lines-----------------------------------




//--------------------------------------------------------------------------------------------------------------------------------------------------------

bool compareLinesByStartPointY(const Vec4i &a, const Vec4i &b) {
    // Compare by the y-coordinate of the start point
    return a[1] < b[1];
}

double distance(const Point& p1, const Point& p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

//to compare lines by their y position
vector<Vec4i> filterLinesByDistance2(vector<Vec4i>& lines, double minDistance) {
    vector<Vec4i> filteredLines;
    if (lines.empty()) return filteredLines;

    filteredLines.push_back(lines[0]);
    Point lastMidpoint = findmidpoint(lines[0]);

    for (size_t i = 1; i < lines.size(); ++i) {
        Point currentMidpoint = findmidpoint(lines[i]);
        if (distance(lastMidpoint, currentMidpoint) >= minDistance) {
            filteredLines.push_back(lines[i]);
            lastMidpoint = currentMidpoint;
        }
    }

    return filteredLines;
}
//---------------------------------------------------------------------to filter by distance of midpoints



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
        // imshow("colorxyz"+ to_string(i), imgxyz);
        
        // ------------------------------------------------------applying masking and gamma to xyz to the image------------------------------
        masked = masking(imgxyz,i);
        //
        float gamma = 3.0; //3.2
        cv::Mat gammaresult;
        gammaCorrection(masked, gammaresult, gamma);
        // imshow("Gamma corrected "+ to_string(i), gammaresult); 




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
        // imshow("gray"+ to_string(i), gray);
        
        //-------------------------------------------------------applying morphological operation-----------------------------------
        morphologyEx( gray, final,MORPH_GRADIENT, Mat());
        // imshow("Morph gradient image" + to_string(i),final);

        
        //-----------------------------------------------------------------applying thresholding------------------------------------
        Mat thresh;
        threshold(final, thresh, 180, 200, THRESH_BINARY + THRESH_OTSU); 
        // imshow("Thresh lines"+ to_string(i), thresh);

                                // -------------------------------------extra thresholding 
                                //     now we use erosion and dilation to make the output more prominent     
                                //     Mat kernel = getStructuringElement(MORPH_CROSS, Size(5, 5));
                                //     // dilate(final,final,kernel);
                                //     dilate(final,final,Mat());
                                //     // imshow("Dilatedimage" + to_string(i),final); 

                                //     erode(final,final,Mat());
                                //     // imshow("Erodedimage" + to_string(i),final);  

                                //     morphologyEx( final, thresh,MORPH_CLOSE, kernel); //closing
                                //     // imshow("After Morph close" + to_string(i),final);
                                //     //-------------------------------------------
        //-----------------------------------------------------------------Canny edge detector-----------------------------------------
        // Apply Canny edge detector
        Mat edges;
        Canny(thresh, edges, 80, 150, 5, true); 
        // imshow("Canny Image"+ to_string(i), edges);


        //-----------------------------------------------------------------Finding Contours--------------------------------------------
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


                    //     vector<vector<Point>> approxContours;
                    //     for (const auto& contour : contours) {
                    //     vector<Point> approx;
                    //     approxPolyDP(contour, approx, 0.0001 * arcLength(contour, true), true);
                    //     approxContours.push_back(approx);
                    //     }

                    // // Draw the approximated polygons on the original image
                    // for (const auto& approx : approxContours) {
                    //     if (approx.size() >= 3) { // Only draw polygons with at least 3 points
                    //         polylines(image, approx, true, Scalar(255, 255, 0), 2, LINE_AA);
                    //     }
                    //     }
                    //     imshow("Polygons", image);

        //---------------------------------------------------------------drawing the contours-----------------------------------------
        Mat contourImg;
        image.copyTo(contourImg);
        for (size_t i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(0, 255, 0); // Green color for contours

        // drawContours(contourImg, contours, (int)i, color, 2, LINE_8, hierarchy, 0); //uncomment to draw
        // drawContours(contourImg, app_contours, (int)i, color, 2, LINE_8, hierarchy, 0);
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
        for (size_t i = 0; i < mergedLines.size(); i++) { //mergedLines has all the lines
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
        //filteredLines has the merged lines.
       


//==============================================================draw parallel lines into rectangles======================================================


        }

        //-------------------------------------------------------checking for negattive slopes and making them positive in the filtered lines---------

        cout<<"size of filtered lines"<< filteredLines.size()<<endl;
        vector<Vec4i> positiveLines = checkPositiveSlope(filteredLines); //works


            //-----------------------------------------------------------using the sort function
         double midDistance = 50;

        vector<Vec4i> filteredLinesre = filterLinesByDistance2(positiveLines, midDistance);

        


        std::sort(positiveLines.begin(), positiveLines.end(), compareLinesByStartPointY);
//-----------------------------------------------------------------------------------------------------------------------------

        vector<Point> midPos;
        for (size_t i = 0; i < positiveLines.size(); i++)
        {
             midPos.push_back(findmidpoint(positiveLines[i]));
        }


        cout<<"size of midpoints"<< midPos.size()<<endl; //works


       
        
        for (size_t i = 0; i < midPos.size()-1; i++)
        {
            Point p = midPos[i];
            Point pn = midPos[i+1];



            Point midomid((p.x+pn.x)/2, (p.y+pn.y)/2); // midpoint of the line connecting the midpoints
        
        //finding the slope
        // Calculate the slope as y2-y1 / x2-x1
        float slope = (pn.y - p.y) / (float)(pn.x - p.x);
        line(randomcolored,p, pn, Scalar(0, 255, 0), 2, LINE_AA);
        putText(randomcolored, format("The angle is %.2f", slope), midomid, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 200, 255), 1, LINE_AA);
        circle(randomcolored, p, 5, Scalar(100, 255, 0), -1);


        }

        for (size_t i = 0; i < filteredLinesre.size()-1; i++)
        {
            Vec4i lin = filteredLinesre[i];

            line(image, Point(lin[0], lin[1]), Point(lin[2], lin[3]), Scalar(0, 255, 0), 2, LINE_AA);
   


        }
     



        //----------------------------------------------------------eliminating the lines which are off angle in the ones that are connected



        imshow("Detected White Lines and Merged", randomcolored);
        int parallelthreshold = 200;
        // Mat filteredRect = constructRectangles(randomcolored,filteredLines, parallelthreshold);
    

    // Display the result
    imshow("Detected White Lines", image);
    imshow("Extended White Lines", extImage);

    imshow("Detected White Lines and Merged", randomcolored);



        i++;
        waitKey(0); 
    }

return (0);

}