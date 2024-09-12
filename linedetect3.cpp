#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load the image
    cv::Mat src = cv::imread("TESTLINESMARKED.jpeg", cv::IMREAD_GRAYSCALE);
    
    if (src.empty()) {
        std::cout << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // Apply edge detection using Canny
    cv::Mat edges;
    cv::Canny(src, edges, 50, 150, 3);

    // Create a vector to store the lines detected
    std::vector<cv::Vec2f> lines;

    // Use Hough Transform to detect lines in the edge-detected image
    cv::HoughLines(edges, lines, 1, CV_PI / 180, 100);

    // Convert to color image to draw the lines
    cv::Mat colorDst;
    cv::cvtColor(edges, colorDst, cv::COLOR_GRAY2BGR);

    // Draw the lines
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 500 * (-b));
        pt1.y = cvRound(y0 + 500 * (a));
        pt2.x = cvRound(x0 - 500 * (-b));
        pt2.y = cvRound(y0 - 500 * (a));
        cv::line(colorDst, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    // Show the result
    cv::imshow("Detected Lines", colorDst);
    cv::waitKey(0);

    return 0;
}
