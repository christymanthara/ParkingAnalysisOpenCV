#include <opencv2/opencv.hpp>
#include <vector>
#include "tinyxml2.h"

using namespace tinyxml2;
using namespace std;

int main(){
    XMLDocument xmlDoc;
    XMLError eResult = xmlDoc.LoadFile("/home/ms/parking-space/data/sequence0/bounding_boxes/2013-02-24_10_05_04.xml");
    if (eResult != XML_SUCCESS) {
        std::cerr << "Error loading XML file!" << std::endl;
        return -1;
    }
    else{
        std::cout<<"XML file loaded"<<std::endl;
    }

    XMLElement *parkingElement = xmlDoc.FirstChildElement("parking");
    if(parkingElement == nullptr){
        std::cerr<<"No <parking> element found!!"<<std::endl;
        return -1;
    }

    for (XMLElement *space = parkingElement->FirstChildElement("space"); space != nullptr; space = space->NextSiblingElement("space")) {
        XMLElement *rotatedRect = space->FirstChildElement("rotatedRect");

        // Extract center coordinates
        XMLElement *center = rotatedRect->FirstChildElement("center");
        float centerX = center->FloatAttribute("x");
        float centerY = center->FloatAttribute("y");
        std::cout << "  Center: (" << centerX << ", " << centerY << ")" << std::endl;

        // Extract size (width and height)
        XMLElement *size = rotatedRect->FirstChildElement("size");
        float width = size->FloatAttribute("w");
        float height = size->FloatAttribute("h");
        std::cout << "  Size: (" << width << " x " << height << ")" << std::endl;

        // Extract rotation angle
        XMLElement *angle = rotatedRect->FirstChildElement("angle");
        float rotationAngle = angle->FloatAttribute("d");
         std::cout << "  Angle: " << rotationAngle << " degrees" << std::endl;

    }
    return 0;
}