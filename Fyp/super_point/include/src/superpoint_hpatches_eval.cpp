#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <fstream>
#include <sstream>
#include <glog/logging.h>
#include "./superpoint.hpp"


using namespace std;
using namespace cv;



Mat readHomographyFile(const string& filename) {
    Mat H(3, 3, CV_64F);
    ifstream file(filename);
    if (!file.is_open()) {
        LOG(ERROR) << "Cannot open homography file: " << filename;
        return Mat();
    }
    
    string line;
    for (int i = 0; i < 3; i++) {
        if (!getline(file, line)) {
            LOG(ERROR) << "Failed to read line " << i << " from homography file";
            return Mat();
        }
        istringstream iss(line);
        for (int j = 0; j < 3; j++) {
            if (!(iss >> H.at<double>(i, j))) {
                LOG(ERROR) << "Failed to parse homography value at position " << i << "," << j;
                return Mat();
            }
        }
    }
    LOG(INFO) << "Successfully read homography matrix:\n" << H;
    return H;
}

struct EvaluationMetrics {
    double repeatability;
    double localization_error;
    double matching_score;
    int num_detected_keypoints;
    double precision;
    double recall;
};

// Function to calculate distance between two points
double pointDistance(const Point2f& p1, const Point2f& p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// Function to match descriptors between two images
vector<DMatch> matchDescriptors(const vector<vector<float>>& desc1, 
                              const vector<vector<float>>& desc2,
                              float threshold = 0.7) {
    vector<DMatch> matches;
    
    for (size_t i = 0; i < desc1.size(); i++) {
        float best_dist = FLT_MAX;
        float second_best_dist = FLT_MAX;
        int best_idx = -1;
        
        // Find the two best matches for descriptor i
        for (size_t j = 0; j < desc2.size(); j++) {
            float dist = 0;
            // Calculate Euclidean distance between descriptors
            for (size_t k = 0; k < desc1[i].size(); k++) {
                dist += pow(desc1[i][k] - desc2[j][k], 2);
            }
            dist = sqrt(dist);
            
            if (dist < best_dist) {
                second_best_dist = best_dist;
                best_dist = dist;
                best_idx = j;
            } else if (dist < second_best_dist) {
                second_best_dist = dist;
            }
        }
        
        // Apply Lowe's ratio test
        if (best_dist < threshold * second_best_dist) {
            DMatch match;
            match.queryIdx = i;
            match.trainIdx = best_idx;
            match.distance = best_dist;
            matches.push_back(match);
        }
    }
    
    return matches;
}

// Change the function signature to accept unique_ptr by reference
EvaluationMetrics evaluateSuperPoint(const std::unique_ptr<vitis::ai::SuperPoint>& superpoint,
    const Mat& img1,
    const Mat& img2,
    const Mat& H,
    double distance_threshold = 3.0) {
EvaluationMetrics metrics;

// Run SuperPoint on both images
vector<Mat> imgs1{img1};
vector<Mat> imgs2{img2};

auto results1 = superpoint->run(imgs1);
auto results2 = superpoint->run(imgs2);

// Get keypoints and descriptors
auto kpts1 = results1[0].keypoints;
auto desc1 = results1[0].descriptor;
auto kpts2 = results2[0].keypoints;
auto desc2 = results2[0].descriptor;

// Convert keypoints to OpenCV format
vector<Point2f> cv_kpts1, cv_kpts2;
for (const auto& kp : kpts1) cv_kpts1.emplace_back(kp.first, kp.second);
for (const auto& kp : kpts2) cv_kpts2.emplace_back(kp.first, kp.second);

// Transform keypoints from image 1 to image 2 using homography
vector<Point2f> transformed_kpts1;
perspectiveTransform(cv_kpts1, transformed_kpts1, H);

// Count correct matches (repeatability)
int num_correct = 0;
double total_error = 0;

for (size_t i = 0; i < transformed_kpts1.size(); i++) {
double min_dist = DBL_MAX;
for (const auto& kp2 : cv_kpts2) {
double dist = pointDistance(transformed_kpts1[i], kp2);
min_dist = min(min_dist, dist);
}
if (min_dist < distance_threshold) {
num_correct++;
total_error += min_dist;
}
}

// Calculate repeatability
metrics.repeatability = static_cast<double>(num_correct) / 
min(cv_kpts1.size(), cv_kpts2.size());

// Calculate average localization error
metrics.localization_error = num_correct > 0 ? total_error / num_correct : 0;

// Match descriptors
auto matches = matchDescriptors(desc1, desc2);

// Calculate matching score
metrics.matching_score = static_cast<double>(matches.size()) / 
min(desc1.size(), desc2.size());

// Calculate precision and recall
int true_positives = 0;
for (const auto& match : matches) {
Point2f transformed_pt = transformed_kpts1[match.queryIdx];
Point2f matched_pt = cv_kpts2[match.trainIdx];

if (pointDistance(transformed_pt, matched_pt) < distance_threshold) {
true_positives++;
}
}

metrics.precision = matches.size() > 0 ? 
static_cast<double>(true_positives) / matches.size() : 0;
metrics.recall = static_cast<double>(true_positives) / cv_kpts1.size();
metrics.num_detected_keypoints = cv_kpts1.size();

return metrics;
}

int main(int argc, char* argv[]) {
    // Initialize Google logging
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;  // Log to console
    // Enable debug logging
    setenv("DEBUG_SUPERPOINT", "1", 1);
    if (argc < 3) {
        LOG(ERROR) << "Usage: " << argv[0] << " model_name hpatches_sequence_path";
        return -1;
    }

    string model_name = argv[1];
    string sequence_path = argv[2];

    LOG(INFO) << "Creating SuperPoint instance with model: " << model_name;
    auto superpoint = vitis::ai::SuperPoint::create(model_name, 1);
    if (!superpoint) {
        LOG(ERROR) << "Failed to create SuperPoint instance";
        return -1;
    }
    LOG(INFO) << "SuperPoint instance created successfully";

    // Get model input dimensions
    int input_width = superpoint->getInputWidth();
    int input_height = superpoint->getInputHeight();
    LOG(INFO) << "SuperPoint model input dimensions: " << input_width << "x" << input_height;


    // Load reference image (1.ppm)
    string ref_img_path = sequence_path + "/1.ppm";
    LOG(INFO) << "Loading reference image: " << ref_img_path;
    Mat img1 = imread(ref_img_path, IMREAD_COLOR);


    if (img1.empty()) {
        LOG(ERROR) << "Failed to load reference image: " << ref_img_path;
        return -1;
    }
    LOG(INFO) << "Reference image loaded successfully. Size: " << img1.size();

    // Load second image (2.ppm)
    string img2_path = sequence_path + "/2.ppm";
    LOG(INFO) << "Loading second image: " << img2_path;
    Mat img2 = imread(img2_path, IMREAD_COLOR);
    if (img2.empty()) {
        LOG(ERROR) << "Failed to load second image: " << img2_path;
        return -1;
    }
    LOG(INFO) << "Second image loaded successfully. Size: " << img2.size();

    // Load homography matrix
    string h_path = sequence_path + "/H_1_2";
    LOG(INFO) << "Loading homography matrix: " << h_path;
    Mat H = readHomographyFile(h_path);
    if (H.empty()) {
        LOG(ERROR) << "Failed to load homography matrix";
        return -1;
    }
    
    // Process both images
    LOG(INFO) << "Processing image pair...";
    try {
        // Run SuperPoint on both images
        vector<Mat> batch1{img1};
        vector<Mat> batch2{img2};
        
        LOG(INFO) << "Running SuperPoint on first image...";
        LOG(INFO) << "Input batch size: " << batch1.size();
        for (size_t i = 0; i < batch1.size(); ++i) {
            LOG(INFO) << "Image " << i << " size: " << batch1[i].size();
        }
        auto results1 = superpoint->run(batch1);
        auto kpts1 = results1[0].keypoints;
        auto desc1 = results1[0].descriptor;
        LOG(INFO) << "Found " << kpts1.size() << " keypoints in first image";
        LOG(INFO) << "Descriptor dimensions: " << desc1.size() << " descriptors, each with " 
                  << (desc1.size() > 0 ? desc1[0].size() : 0) << " dimensions";
        
        // Print sample descriptor for first keypoint
        if (!desc1.empty() && !desc1[0].empty()) {
            LOG(INFO) << "Sample descriptor for first keypoint:";
            stringstream ss;
            ss << "[";
            for (size_t i = 0; i < min(size_t(5), desc1[0].size()); i++) {
                ss << desc1[0][i] << ", ";
            }
            ss << "...]";
            LOG(INFO) << ss.str();
        }
        
        // After running SuperPoint on first image
        LOG(INFO) << "Keypoint distribution:";
        int quadrants[4] = {0, 0, 0, 0}; // topleft, topright, bottomleft, bottomright
        for (const auto& kp : kpts1) {
            int x = kp.first;
            int y = kp.second;
            int half_width = img1.cols / 2;
            int half_height = img1.rows / 2;
            
            if (x < half_width && y < half_height) quadrants[0]++;
            else if (x >= half_width && y < half_height) quadrants[1]++;
            else if (x < half_width && y >= half_height) quadrants[2]++;
            else quadrants[3]++;
        }
        
        LOG(INFO) << "Top-left: " << quadrants[0] << " keypoints (" 
                << 100.0 * quadrants[0] / kpts1.size() << "%)";
        LOG(INFO) << "Top-right: " << quadrants[1] << " keypoints (" 
                << 100.0 * quadrants[1] / kpts1.size() << "%)";
        LOG(INFO) << "Bottom-left: " << quadrants[2] << " keypoints (" 
                << 100.0 * quadrants[2] / kpts1.size() << "%)";
        LOG(INFO) << "Bottom-right: " << quadrants[3] << " keypoints (" 
                << 100.0 * quadrants[3] / kpts1.size() << "%)";
            
            // Save debug visualization of keypoints
            Mat keypoint_viz = img1.clone();
            for (const auto& kp : kpts1) {
                circle(keypoint_viz, Point(kp.first, kp.second), 3, Scalar(0, 255, 0), -1);
            }
            string kp_vis_path = sequence_path + "/keypoints_visualization.jpg";
            imwrite(kp_vis_path, keypoint_viz);
            LOG(INFO) << "Saved keypoint visualization to: " << kp_vis_path;


        LOG(INFO) << "Running SuperPoint on second image...";
        auto results2 = superpoint->run(batch2);
        auto kpts2 = results2[0].keypoints;
        auto desc2 = results2[0].descriptor;
        LOG(INFO) << "Found " << kpts2.size() << " keypoints in second image";
        LOG(INFO) << "Descriptor dimensions: " << desc2.size() << " descriptors, each with " 
                  << (desc2.size() > 0 ? desc2[0].size() : 0) << " dimensions";
        
        // Match descriptors
        LOG(INFO) << "Matching descriptors between images...";
        auto matches = matchDescriptors(desc1, desc2);
        LOG(INFO) << "Found " << matches.size() << " matches between the images";
        
        // Calculate match ratio
        double match_ratio = static_cast<double>(matches.size()) / min(kpts1.size(), kpts2.size());
        LOG(INFO) << "Match ratio: " << match_ratio * 100 << "%";
        
        // Evaluate the SuperPoint performance
        LOG(INFO) << "Evaluating SuperPoint performance...";
        auto metrics = evaluateSuperPoint(superpoint, img1, img2, H);
        
        LOG(INFO) << "==== SuperPoint Evaluation Results ====";
        LOG(INFO) << "Number of keypoints detected: " << metrics.num_detected_keypoints;
        LOG(INFO) << "Repeatability: " << metrics.repeatability * 100 << "%";
        LOG(INFO) << "Localization error: " << metrics.localization_error << " pixels";
        LOG(INFO) << "Matching score: " << metrics.matching_score * 100 << "%";
        LOG(INFO) << "Precision: " << metrics.precision * 100 << "%";
        LOG(INFO) << "Recall: " << metrics.recall * 100 << "%";
        
        // Visualize matches (optional, requires display)
        // Convert keypoints to OpenCV format for visualization
        vector<KeyPoint> cv_keypoints1, cv_keypoints2;
        for (const auto& kp : kpts1) {
            cv_keypoints1.emplace_back(kp.first, kp.second, 1.0);
        }
        for (const auto& kp : kpts2) {
            cv_keypoints2.emplace_back(kp.first, kp.second, 1.0);
        }
        
        // Draw matches
        Mat img_matches;
        drawMatches(img1, cv_keypoints1, img2, cv_keypoints2, matches, img_matches);
        
        // Save the visualization
        string output_path = sequence_path + "/superpoint_matches.jpg";
        imwrite(output_path, img_matches);
        LOG(INFO) << "Saved match visualization to: " << output_path;
        
    } catch (const std::exception& e) {
        LOG(ERROR) << "Exception while processing image pair: " << e.what();
        return -1;
    }

    LOG(INFO) << "Program completed successfully";
    return 0;
}