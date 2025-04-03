#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include "./superpoint.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    cout << "Starting SuperPoint execution for throughput measurement." << endl;

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " model_name image" << endl;
        return -1;
    }

    string model_name = argv[1];
    cout << "Model name: " << model_name << endl;

    Mat img = imread(argv[2], cv::IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Error: Failed to load image: " << argv[2] << endl;
        return -1;
    }
    cout << "Image loaded successfully." << endl;

    auto superpoint = vitis::ai::SuperPoint::create(model_name, 5);
    if (!superpoint) {
        cerr << "Error: Failed to create SuperPoint instance." << endl;
        return -1;
    }
    cout << "SuperPoint model created successfully." << endl;

    int batch_size = superpoint->get_input_batch();
    vector<Mat> imgs(batch_size, img); // Fill vector with same image

    int total_images = 1000;  // Number of images to process
    int iterations = total_images / batch_size; // Number of iterations
    cout << "Processing " << total_images << " images in " << iterations << " iterations." << endl;

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        auto result = superpoint->run(imgs);
    }

    auto end = chrono::high_resolution_clock::now();
    double total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count() / 1000.0; // in seconds

    double throughput = total_images / total_time;
    cout << "Processing complete." << endl;
    cout << "Total images processed: " << total_images << endl;
    cout << "Total time: " << total_time << " seconds" << endl;
    cout << "Throughput: " << throughput << " images per second" << endl;

    return 0;
}
