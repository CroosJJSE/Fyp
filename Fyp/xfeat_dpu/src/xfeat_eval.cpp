//xfeat_eval.cpp
#include "XFextractorDPU.h"
#include <opencv2/features2d.hpp>
#include <fstream>

using namespace ORB_SLAM3;
using namespace std;
cv::Mat readHomography(const std::string& path) {
    cv::Mat H(3, 3, CV_64F);
    std::ifstream file(path);
    for(int i=0; i<3; ++i)
        for(int j=0; j<3; ++j)
            file >> H.at<double>(i,j);
    return H;
}

double computeRepeatability(const std::vector<cv::KeyPoint>& kpts1,
                          const std::vector<cv::KeyPoint>& kpts2,
                          const cv::Mat& H, 
                          float thresh=3.0) {
    std::vector<cv::Point2f> pts1, pts2;
    for(const auto& kp : kpts1) pts1.push_back(kp.pt);
    for(const auto& kp : kpts2) pts2.push_back(kp.pt);
    
    cv::perspectiveTransform(pts1, pts1, H);
    
    int matches = 0;
    for(const auto& pt1 : pts1) {
        for(const auto& pt2 : pts2) {
            if(cv::norm(pt1 - pt2) < thresh) {
                matches++;
                break;
            }
        }
    }
    return static_cast<double>(matches)/std::min(kpts1.size(), kpts2.size());
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    
    if(argc < 4) {
        LOG(ERROR) << "Usage: " << argv[0] << " <model.xmodel> <img1> <img2> <H_file>";
        return -1;
    }
    
    XFextractorDPU extractor(argv[1]);
    cv::Mat img1 = cv::imread(argv[2]);
    cv::Mat img2 = cv::imread(argv[3]);
    cv::Mat H = readHomography(argv[4]);
    
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat desc1, desc2;
    std::vector<int> lapping;
    
    extractor(img1, kpts1, desc1, lapping);
    extractor(img2, kpts2, desc2, lapping);
    
    // Evaluation metrics
    double rep = computeRepeatability(kpts1, kpts2, H);
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);
    
    cout << "==== XFeat DPU Results ====" << endl ;
    cout << "Keypoints: " << kpts1.size() << " | " << kpts2.size() << endl ;
    cout << "Repeatability: " << rep*100 << "%" << endl;
    cout << "Matches: " << matches.size() << endl;
    
    return 0;
}