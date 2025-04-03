#include "SuperPointExtractor.h"
#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{
SuperPointExtractor::SuperPointExtractor(int nfeatures, float scaleFactor, int nlevels,
                                         int iniThFAST, int minThFAST,
                                         const std::string& model_name, int num_runners)
    : XFextractor(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST)
{
    superpoint = vitis::ai::SuperPoint::create(model_name, num_runners);
}

SuperPointExtractor::~SuperPointExtractor()
{
}

int SuperPointExtractor::operator()(cv::InputArray _image, cv::InputArray _mask,
                                    std::vector<cv::KeyPoint>& _keypoints,
                                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea)
{
    cv::Mat image = _image.getMat();
    std::vector<vitis::ai::SuperPointResult> results = superpoint->run({image});
    if (results.empty()) {
        std::cerr << "No results from SuperPoint extractor." << std::endl;
        return 0;
    }
    const auto& result = results[0];
    _keypoints.clear();
    for (size_t i = 0; i < std::min(result.keypoints.size(), static_cast<size_t>(nfeatures)); ++i) {
        const auto& kp = result.keypoints[i];
        cv::KeyPoint cv_kp;
        cv_kp.pt.x = kp.first;
        cv_kp.pt.y = kp.second;
        cv_kp.size = 1.0;
        cv_kp.angle = -1;
        cv_kp.response = 1.0;
        cv_kp.octave = 0;
        _keypoints.push_back(cv_kp);
    }
    if (!_mask.empty()) {
        cv::Mat mask = _mask.getMat();
        std::vector<cv::KeyPoint> filtered_keypoints;
        for (const auto& kp : _keypoints) {
            if (mask.at<uchar>(static_cast<int>(kp.pt.y), static_cast<int>(kp.pt.x)) != 0) {
                filtered_keypoints.push_back(kp);
            }
        }
        _keypoints = filtered_keypoints;
    }
    if (_descriptors.needed()) {
        _descriptors.create(_keypoints.size(), 256, CV_32F);
        cv::Mat desc = _descriptors.getMat();
        for (size_t i = 0; i < _keypoints.size(); ++i) {
            const auto& desc_i = result.descriptor[i];
            float* rowPtr = desc.ptr<float>(i);
            for (size_t j = 0; j < desc_i.size(); ++j) {
                rowPtr[j] = desc_i[j];
            }
        }
    }
    vLappingArea.clear();
    return _keypoints.size();
}
}