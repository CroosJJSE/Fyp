#include "XFextractorDPU.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>

using namespace cv;
using namespace std;
using namespace std::chrono;

// Throughput test parameters
const int WARMUP_RUNS = 5;
const int BENCHMARK_RUNS = 100;
const string OUTPUT_CSV = "throughput_results.csv";

void runBenchmark(const string& model_path, const string& image_path) {
    // Initialize extractor
    ORB_SLAM3::XFextractorDPU extractor(model_path);
    
    // Load test image
    Mat img = imread(image_path);
    if(img.empty()) {
        cerr << "Error: Could not load image: " << image_path << endl;
        return;
    }

    vector<KeyPoint> keypoints;
    Mat descriptors;
    vector<int> lapping = {0, img.cols};
    
    // Warmup runs
    cout << "Running warmup..." << endl;
    for(int i = 0; i < WARMUP_RUNS; i++) {
        extractor(img, keypoints, descriptors, lapping);
    }

    // Benchmark runs
    cout << "Running benchmark (" << BENCHMARK_RUNS << " iterations)..." << endl;
    vector<double> inference_times;
    double total_time = 0;
    int total_features = 0;

    ofstream csv(OUTPUT_CSV);
    csv << "Run,Time(ms),Features" << endl;

    for(int i = 0; i < BENCHMARK_RUNS; i++) {
        auto start = high_resolution_clock::now();
        extractor(img, keypoints, descriptors, lapping);
        auto end = high_resolution_clock::now();
        
        double elapsed = duration_cast<microseconds>(end - start).count() / 1000.0;
        inference_times.push_back(elapsed);
        total_time += elapsed;
        total_features += keypoints.size();
        
        csv << i+1 << "," << fixed << setprecision(3) << elapsed << "," << keypoints.size() << endl;
    }
    csv.close();

    // Calculate statistics
    sort(inference_times.begin(), inference_times.end());
    double avg_time = total_time / BENCHMARK_RUNS;
    double min_time = inference_times.front();
    double max_time = inference_times.back();
    double median_time = inference_times[BENCHMARK_RUNS/2];
    double fps = 1000.0 / avg_time;
    double avg_features = static_cast<double>(total_features) / BENCHMARK_RUNS;

    // Print results
    cout << "\n==== Throughput Analysis Results ====" << endl;
    cout << "Test image: " << image_path << endl;
    cout << "Image size: " << img.cols << "x" << img.rows << endl;
    cout << "Model: " << model_path << endl;
    cout << "Runs: " << BENCHMARK_RUNS << endl;
    cout << "---------------------------------" << endl;
    cout << "Average features per image: " << avg_features << endl;
    cout << "Average inference time: " << avg_time << " ms" << endl;
    cout << "Minimum inference time: " << min_time << " ms" << endl;
    cout << "Maximum inference time: " << max_time << " ms" << endl;
    cout << "Median inference time: " << median_time << " ms" << endl;
    cout << "Throughput: " << fps << " FPS" << endl;
    cout << "---------------------------------" << endl;
    cout << "Detailed results saved to: " << OUTPUT_CSV << endl;
}

int main(int argc, char** argv) {
    if(argc < 3) {
        cout << "Usage: " << argv[0] << " <model_path> <image_path>" << endl;
        cout << "Example: " << argv[0] << " compiled_by_H_.xmodel 1.ppm" << endl;
        return -1;
    }

    string model_path = argv[1];
    string image_path = argv[2];

    runBenchmark(model_path, image_path);

    return 0;
}