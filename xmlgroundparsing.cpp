// Function to convert rotated rect to a bounding box (cv::Rect)
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "pugixml.hpp"

using namespace std;
using namespace cv;

cv::Rect rotatedRectToBoundingBox(const RotatedRect& rRect) {
    return rRect.boundingRect();
}

// Function to parse the XML and extract ground truth rectangles
std::vector<cv::Rect> parseXMLGroundTruth(const std::string& xmlPath) {
    std::vector<cv::Rect> groundTruthRects;

    // Load the XML file
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(xmlPath.c_str());

    if (!result) {
        cerr << "Failed to load XML file: " << result.description() << endl;
        return groundTruthRects;
    }

    // Get the root node (<parking>)
    pugi::xml_node parking = doc.child("parking");
    if (!parking) {
        cerr << "Invalid XML structure: <parking> not found." << endl;
        return groundTruthRects;
    }

    // Iterate over each <space> element
    for (pugi::xml_node space : parking.children("space")) {
        pugi::xml_node rotatedRect = space.child("rotatedRect");
        if (!rotatedRect) continue;

        // Extract <center>, <size>, and <angle>
        pugi::xml_node center = rotatedRect.child("center");
        pugi::xml_node size = rotatedRect.child("size");
        pugi::xml_node angle = rotatedRect.child("angle");

        if (!center || !size || !angle) continue;

        // Read values from XML
        int centerX = center.attribute("x").as_int();
        int centerY = center.attribute("y").as_int();
        int width = size.attribute("w").as_int();
        int height = size.attribute("h").as_int();
        float angleDeg = angle.attribute("d").as_float();

        // Create the RotatedRect
        Point2f centerPoint(centerX, centerY);
        Size2f rectSize(width, height);
        RotatedRect rotatedRectObj(centerPoint, rectSize, angleDeg);

        // Convert the rotated rectangle to a bounding box
        cv::Rect boundingBox = rotatedRectToBoundingBox(rotatedRectObj);
        groundTruthRects.push_back(boundingBox);
    }

    return groundTruthRects;
}