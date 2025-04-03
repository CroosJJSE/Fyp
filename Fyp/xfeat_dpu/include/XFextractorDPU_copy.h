#ifndef XFEXTRACTOR_DPU_H
#define XFEXTRACTOR_DPU_H

#include "XfeatSLAMExtractor.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <glog/logging.h>

namespace ORB_SLAM3 {

class XFextractorDPU : public XSLAMFextractor {
public:
    XFextractorDPU(const std::string& model_name, int num_runners=1);
    ~XFextractorDPU();
    
    int operator()(cv::InputArray image, 
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::OutputArray descriptors,
                 std::vector<int>& vLappingArea) override;

    // XFeat-specific interface
    int getInputWidth() const;
    int getInputHeight() const;
    float getScaleFactor() const;

private:
    void preprocess(const cv::Mat& image, std::vector<int8_t>& input_data);
    void postprocess(const std::vector<int8_t>& output_kpts,
                   const std::vector<int8_t>& output_desc,
                   std::vector<cv::KeyPoint>& keypoints,
                   cv::Mat& descriptors);

    std::vector<std::unique_ptr<vitis::ai::DpuTask>> runners_;
    std::vector<vitis::ai::library::InputTensor> input_tensors_;
    std::vector<vitis::ai::library::OutputTensor> output_tensors_;
    
    int input_width_;
    int input_height_;
    float scale_factor_;
    float conf_thresh_;
};

} // namespace ORB_SLAM3
#endif