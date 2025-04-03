#include "XFextractorDPU.h"
#include <vitis/ai/profiling.hpp>

namespace ORB_SLAM3 {

XFextractorDPU::XFextractorDPU(const std::string& model_name, 
                             int nfeatures, 
                             float scaleFactor, 
                             int nlevels,
                             int iniThFAST, 
                             int minThFAST,
                             int num_runners)
    : XSLAMFextractor(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST) {
    // Initialize DPU runners
    for(int i = 0; i < num_runners; ++i) {
        runners_.emplace_back(vitis::ai::DpuTask::create(model_name));
    }
    
    // Get tensor info from first runner
    input_tensors_ = runners_[0]->getInputTensor(0u);
    output_tensors_ = runners_[0]->getOutputTensor(0u);
    
    input_width_ = input_tensors_[0].width;
    input_height_ = input_tensors_[0].height;
}

int XFextractorDPU::operator()(cv::InputArray _image, 
                             cv::InputArray _mask,
                             std::vector<cv::KeyPoint>& _keypoints,
                             cv::OutputArray _descriptors, 
                             std::vector<int> &vLappingArea) {
    cv::Mat image = _image.getMat();
    _keypoints.clear();
    
    // 1. Preprocess
    __TIC__(XFEAT_PREPROCESS)
    std::vector<int8_t> input_data;
    preprocess(image, input_data);
    __TOC__(XFEAT_PREPROCESS)

    // 2. DPU Inference
    __TIC__(XFEAT_DPU_RUN)
    auto& runner = runners_[0];
    int8_t* input_ptr = (int8_t*)input_tensors_[0].get_data(0);
    std::memcpy(input_ptr, input_data.data(), input_data.size());
    runner->run(0u);
    __TOC__(XFEAT_DPU_RUN)

    // 3. Postprocess
    __TIC__(XFEAT_POSTPROCESS)
    std::vector<int8_t> kpts_data(output_tensors_[0].size);
    std::vector<int8_t> desc_data(output_tensors_[1].size);
    
    std::memcpy(kpts_data.data(), output_tensors_[0].get_data(0), kpts_data.size());
    std::memcpy(desc_data.data(), output_tensors_[1].get_data(0), desc_data.size());
    
    cv::Mat desc;
    postprocess(kpts_data, desc_data, _keypoints, desc);
    desc.copyTo(_descriptors);
    __TOC__(XFEAT_POSTPROCESS)

    return _keypoints.size();
}

void XFextractorDPU::preprocess(const cv::Mat& image, std::vector<int8_t>& input_data) {
    // Your original preprocess implementation
    cv::Mat gray, resized, float_img;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(input_width_, input_height_));
    
    // Normalization and scaling
    resized.convertTo(float_img, CV_32F, 1.0/255.0);
    float input_scale = vitis::ai::library::tensor_scale(input_tensors_[0]);
    float_img = float_img * input_scale;
    
    // Convert to DPU format
    cv::Mat int8_img;
    float_img.convertTo(int8_img, CV_8SC1);
    input_data.assign((int8_t*)int8_img.data, (int8_t*)int8_img.data + int8_img.total());
}

void XFextractorDPU::postprocess(const std::vector<int8_t>& output_kpts,
                               const std::vector<int8_t>& output_desc,
                               std::vector<cv::KeyPoint>& keypoints,
                               cv::Mat& descriptors) {
    // Your original postprocess implementation
    const float kpts_scale = vitis::ai::library::tensor_scale(output_tensors_[0]);
    const float desc_scale = vitis::ai::library::tensor_scale(output_tensors_[1]);
    
    // Keypoint extraction logic
    const int grid_size = sqrt(output_kpts.size()/65);
    for(int y=0; y<grid_size; y++) {
        for(int x=0; x<grid_size; x++) {
            float max_response = -1;
            int max_channel = -1;
            
            for(int c=0; c<64; c++) {
                float response = output_kpts[y*grid_size*65 + x*65 + c] * kpts_scale;
                if(response > max_response) {
                    max_response = response;
                    max_channel = c;
                }
            }
            
            if(max_response > conf_thresh_) {
                cv::KeyPoint kp;
                kp.pt.x = x * 8 + (max_channel % 8);
                kp.pt.y = y * 8 + (max_channel / 8);
                kp.response = max_response;
                kp.size = 8.0f;
                keypoints.push_back(kp);
            }
        }
    }
    
    // Descriptor processing
    descriptors.create(keypoints.size(), 256, CV_32F);
    for(size_t i=0; i<keypoints.size(); i++) {
        const auto& kp = keypoints[i];
        int grid_x = kp.pt.x / 8;
        int grid_y = kp.pt.y / 8;
        
        float* desc = descriptors.ptr<float>(i);
        for(int d=0; d<256; d++) {
            desc[d] = output_desc[grid_y*grid_size*256 + grid_x*256 + d] * desc_scale;
        }
        cv::normalize(descriptors.row(i), descriptors.row(i)); // L2 normalize
    }
}

} // namespace ORB_SLAM3