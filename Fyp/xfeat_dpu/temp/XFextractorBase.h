// XFextractorBase.h
#ifndef XFEXTRACTOR_BASE_H
#define XFEXTRACTOR_BASE_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace ORB_SLAM3 {

class XFextractorBase {
public:
    virtual ~XFextractorBase() = default;

    // Define the interface for feature extraction
    virtual int operator()(cv::InputArray _image, cv::InputArray _mask,
                           std::vector<cv::KeyPoint>& _keypoints,
                           cv::OutputArray _descriptors, std::vector<int>& vLappingArea) = 0;

    virtual int GetLevels() = 0;
    virtual float GetScaleFactor() = 0; //implemented
    virtual std::vector<float> GetScaleFactors() = 0;
    virtual std::vector<float> GetInverseScaleFactors() = 0;
    virtual std::vector<float> GetScaleSigmaSquares() = 0;
    virtual std::vector<float> GetInverseScaleSigmaSquares() = 0;
};

} // namespace ORB_SLAM3

#endif // XFEXTRACTOR_BASE_H