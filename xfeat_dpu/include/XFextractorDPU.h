#ifndef XFEXTRACTOR_DPU_H
#define XFEXTRACTOR_DPU_H

#include "XfeatSLAMExtractor.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/library/tensor.hpp>

namespace ORB_SLAM3 {

class XFextractorDPU : public XSLAMFextractor {
public:
    XFextractorDPU(const std::string& model_name, 
                 int nfeatures = 1000, 
                 float scaleFactor = 1.2f, 
                 int nlevels = 8,
                 int iniThFAST = 20, 
                 int minThFAST = 7,
                 int num_runners = 1);
    
    ~XFextractorDPU() = default;
    
    // Main virtual function implementation
    int operator()(cv::InputArray _image, cv::InputArray _mask,
                 std::vector<cv::KeyPoint>& _keypoints,
                 cv::OutputArray _descriptors, 
                 std::vector<int> &vLappingArea) override;

    // Backward-compatible overload
    int operator()(cv::InputArray _image, 
                 std::vector<cv::KeyPoint>& _keypoints,
                 cv::OutputArray _descriptors,
                 std::vector<int> &vLappingArea) {
        cv::Mat dummy_mask;
        return this->operator()(_image, dummy_mask, _keypoints, _descriptors, vLappingArea);
    }

    // DPU-specific accessors (inline in header)
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }

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
    float conf_thresh_ = 0.015f;
};

} // namespace ORB_SLAM3
#endif