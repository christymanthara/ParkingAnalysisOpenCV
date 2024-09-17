#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "utilities.hpp"


void ensurePositiveSlope(Point& pt1, Point& pt2) 
{
    // Calculate the slope
    float slope = (pt2.y - pt1.y) / (float)(pt2.x - pt1.x);

    // If the slope is negative, swap the points
    if (slope < 0) {
        swap(pt1, pt2);
    }
}

int main() {
    // Define two points for the line
    Point pt1(10, 20);
    Point pt2(30, 10);

    // Ensure the line has a positive slope
    ensurePositiveSlope(pt1, pt2);

    cout << "New coordinates after ensuring positive slope: (" 
         << pt1.x << ", " << pt1.y << "), (" << pt2.x << ", " << pt2.y << ")" << endl;

    return 0;
}




vector<pair<Vec4i, Vec4i>> closestLinePairs;

    float minDistance = numeric_limits<float>::max();

    for (size_t i = 0; i < midPos.size(); ++i) {
        for (size_t j = i + 1; j < midPos.size(); ++j) {
            float dist = distance(midPos[i], midPos[j]);
            if (dist < minDistance) {
                minDistance = dist;
                closestLinePairs.clear();
                closestLinePairs.push_back(make_pair(positiveLines[i], positiveLines[j]));
            } else if (dist == minDistance) {
                closestLinePairs.push_back(make_pair(positiveLines[i], positiveLines[j]));
            }
        }
    }

    // Draw the closest line pairs
    for (const auto& linePair : closestLinePairs) {
        line(image, findMidpoint(linePair.first), findMidpoint(linePair.second), Scalar(0, 255, 0), 2, LINE_AA);
    }

    