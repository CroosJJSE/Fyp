

#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <memory>
#include <vector>
#include <utility>


namespace vitis {
namespace ai {
    
  struct SuperPointResult {
    size_t index;  // To keep track of the image order
    std::vector<std::pair<float, float>> keypoints;
    std::vector<std::vector<float>> descriptor;
    float scale_w;
    float scale_h;
  };

  class SuperPoint {
    public:
      static std::unique_ptr<SuperPoint> create(const std::string& model_name, int num_runners);

    protected:
      explicit SuperPoint(const std::string& model_name, int num_runners);
      SuperPoint(const SuperPoint&) = delete;
      SuperPoint& operator=(const SuperPoint&) = delete;

    public:
      virtual ~SuperPoint();
      virtual std::vector<SuperPointResult> run(const std::vector<cv::Mat>& imgs) = 0;
      virtual size_t get_input_batch() = 0;
      virtual int getInputWidth() const = 0;
      virtual int getInputHeight() const = 0;
    };
  }
}