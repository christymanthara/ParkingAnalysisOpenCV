#include <iostream>
#include <opencv2/opencv.hpp>
#include <pugixml.hpp>
#include <vector>

using namespace std;
using namespace cv;

int main() {
    
    vector<vector<Point>> contours;
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file("ParkingLot_dataset/sequence3/bounding_boxes/2013-03-19_07_25_01.xml");

    if (!result) {
        cout << "Failed to load XML file: " << result.description() << endl;
        //return contours;
    }
    

    else 
    {

        cout<<"xml file succcessfully loaded \n";
    
    }
    return 0;
}