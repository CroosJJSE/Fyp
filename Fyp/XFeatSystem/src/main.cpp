#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "SuperPointExtractor.h"

int main(int argc, char* argv[]) {
    // Check for command-line argument (image path)
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    // Load the image from the provided path
    std::string image_path = argv[1];
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }

    // Parameters for SuperPointExtractor
    int nfeatures = 1000;       // Maximum number of features to extract
    float scaleFactor = 1.2f;   // Pyramid scale factor
    int nlevels = 8;            // Number of pyramid levels
    int iniThFAST = 20;         // Initial FAST threshold
    int minThFAST = 7;          // Minimum FAST threshold
    std::string model_name = "superpoint_model";  // Name of the SuperPoint model (adjust as needed)
    int num_runners = 1;        // Number of DPU runners (adjust for Kria KR260 setup)

    // Initialize the SuperPointExtractor
    ORB_SLAM3::SuperPointExtractor extractor(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST, model_name, num_runners);

    // Prepare an empty mask (optional, can be used if needed)
    cv::Mat mask;

    // Variables to store the output
    std::vector<cv::KeyPoint> keypoints;  // Keypoint locations and properties
    cv::Mat descriptors;                  // Descriptor matrix
    std::vector<int> vLappingArea;        // Lapping area (not used here, but required by the function)

    // Perform feature extraction
    int num_keypoints = extractor(image, mask, keypoints, descriptors, vLappingArea);

    // Output the results to the console
    std::cout << "Number of keypoints detected: " << num_keypoints << std::endl;
    for (size_t i = 0; i < keypoints.size(); ++i) {
        const cv::KeyPoint& kp = keypoints[i];
        std::cout << "Keypoint " << i + 1 << ": (x=" << kp.pt.x << ", y=" << kp.pt.y << ")" << std::endl;
    }

    return 0;
}