#ifndef SUPERPOINT_EXTRACTOR_H
#define SUPERPOINT_EXTRACTOR_H

#include "XFextractor.h"
#include "superpoint.hpp"

namespace ORB_SLAM3
{
class SuperPointExtractor : public XFextractor
{
public:
    SuperPointExtractor(int nfeatures, float scaleFactor, int nlevels,
                        int iniThFAST, int minThFAST,
                        const std::string& model_name, int num_runners);
    ~SuperPointExtractor() override;
    int operator()(cv::InputArray _image, cv::InputArray _mask,
                   std::vector<cv::KeyPoint>& _keypoints,
                   cv::OutputArray _descriptors, std::vector<int> &vLappingArea) override;

private:
    std::unique_ptr<vitis::ai::SuperPoint> superpoint;
};
}

#endif