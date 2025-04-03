#include "XFextractorDPU.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <cfloat>

using namespace cv;
using namespace std;

Mat readHomographyFile(const string& filename) {
    Mat H(3, 3, CV_64F);
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Error: Cannot open homography file: " << filename << endl;
        return Mat();
    }
    
    string line;
    for (int i = 0; i < 3; i++) {
        if (!getline(file, line)) {
            cout << "Error: Failed to read line " << i << " from homography file" << endl;
            return Mat();
        }
        istringstream iss(line);
        for (int j = 0; j < 3; j++) {
            if (!(iss >> H.at<double>(i, j))) {
                cout << "Error: Failed to parse homography value at position " << i << "," << j << endl;
                return Mat();
            }
        }
    }
    return H;
}

double pointDistance(const Point2f& p1, const Point2f& p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

vector<DMatch> matchDescriptors(const Mat& desc1, const Mat& desc2, float threshold = 0.7) {
    vector<DMatch> matches;
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(desc1, desc2, knn_matches, 2);
    
    for(size_t i = 0; i < knn_matches.size(); i++) {
        if(knn_matches[i][0].distance < threshold * knn_matches[i][1].distance) {
            matches.push_back(knn_matches[i][0]);
        }
    }
    return matches;
}

int main(int argc, char** argv) {
    setenv("DEBUG_XFEAT_DPU", "1", 1);
    
    if(argc < 3) {
        cout << "Usage: " << argv[0] << " <model_path> <image_sequence_path>" << endl;
        return -1;
    }

    string model_path = argv[1];
    string sequence_path = argv[2];

    // Create extractor - using correct constructor
    ORB_SLAM3::XFextractorDPU extractor(model_path);

    // Load images
    Mat img1 = imread(sequence_path + "/1.ppm");
    Mat img2 = imread(sequence_path + "/2.ppm");
    Mat H = readHomographyFile(sequence_path + "/H_1_2");

    if(img1.empty() || img2.empty() || H.empty()) {
        cout << "Error: Failed to load input data" << endl;
        return -1;
    }

    // Extract features - using correct operator() signature
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    vector<int> lapping = {0, img1.cols};
    
    extractor(img1, kpts1, desc1, lapping);
    extractor(img2, kpts2, desc2, lapping);

    // Rest of the code remains the same...
    vector<Point2f> pts1, pts2;
    for(const auto& kp : kpts1) pts1.emplace_back(kp.pt);
    for(const auto& kp : kpts2) pts2.emplace_back(kp.pt);
    
    vector<Point2f> transformed_pts1;
    perspectiveTransform(pts1, transformed_pts1, H);

    int num_correct = 0;
    double total_error = 0;
    for(size_t i = 0; i < transformed_pts1.size(); i++) {
        double min_dist = DBL_MAX;
        for(const auto& pt : pts2) {
            min_dist = min(min_dist, pointDistance(transformed_pts1[i], pt));
        }
        if(min_dist < 3.0) {
            num_correct++;
            total_error += min_dist;
        }
    }
    double repeatability = static_cast<double>(num_correct) / min(pts1.size(), pts2.size());
    
    auto matches = matchDescriptors(desc1, desc2);
    double matching_score = static_cast<double>(matches.size()) / min(desc1.rows, desc2.rows);

    cout << "==== XFeat DPU Results ====" << endl;
    cout << "Keypoints (img1/img2): " << kpts1.size() << "/" << kpts2.size() << endl;
    cout << "Repeatability: " << repeatability * 100 << "%" << endl;
    cout << "Avg. Loc. Error: " << (num_correct > 0 ? total_error/num_correct : 0) << "px" << endl;
    cout << "Matching Score: " << matching_score * 100 << "%" << endl;

    Mat out_img;
    drawKeypoints(img1, kpts1, out_img);
    imwrite("xfeat_kpts.jpg", out_img);
    
    Mat match_img;
    vector<KeyPoint> cv_kpts1(kpts1.begin(), kpts1.end());
    vector<KeyPoint> cv_kpts2(kpts2.begin(), kpts2.end());
    drawMatches(img1, cv_kpts1, img2, cv_kpts2, matches, match_img);
    imwrite("xfeat_matches.jpg", match_img);

    return 0;
}