

#include "superpoint.hpp"

#include <glog/logging.h>
#include <memory>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <queue>
#include <atomic>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/library/tensor.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/ai/math.hpp>

#define HW_SOFTMAX
//#define ENABLE_NEON

DEF_ENV_PARAM(DEBUG_SUPERPOINT, "0");
DEF_ENV_PARAM(DUMP_SUPERPOINT, "0");

using namespace std;
using namespace cv;

static vector<vitis::ai::library::OutputTensor> sort_tensors(
    const vector<vitis::ai::library::OutputTensor>& tensors,
    vector<size_t>& chas) {
  vector<vitis::ai::library::OutputTensor> ordered_tensors;
  for (auto i = 0u; i < chas.size(); ++i)
    for (auto j = 0u; j < tensors.size(); ++j)
      if (tensors[j].channel == chas[i]) {
        ordered_tensors.push_back(tensors[j]);
        LOG_IF(INFO, ENV_PARAM(DEBUG_SUPERPOINT))
          << "tensor name: " << tensors[j].name;
        break;
      }
  return ordered_tensors;
}

namespace vitis {
namespace ai {

// Thread-safe queue implementation
template <typename T>
class ThreadSafeQueue {
 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cond_var_;
  bool shutdown_;

 public:
  ThreadSafeQueue() : shutdown_(false) {}

  void enqueue(const T& item) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(item);
    }
    cond_var_.notify_one();
  }

  bool dequeue(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_var_.wait(lock, [this]() { return !queue_.empty() || shutdown_; });
    if (shutdown_ && queue_.empty()) {
      return false;
    }
    item = queue_.front();
    queue_.pop();
    return true;
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    cond_var_.notify_all();
  }
};

// Data structures for pipeline stages
struct DpuInferenceTask {
  size_t index;
  std::vector<int8_t> input_data;
  float scale_w;
  float scale_h;
};

struct DpuInferenceResult {
  size_t index;
  std::vector<int8_t> output_data1;
  std::vector<int8_t> output_data2;
  float scale_w;
  float scale_h;
  float scale1;
  float scale2;
};

class SuperPointImp : public SuperPoint {
 public:
  SuperPointImp(const std::string& model_name, int num_runners);

 public:
  virtual ~SuperPointImp();
  virtual std::vector<SuperPointResult> run(const std::vector<cv::Mat>& imgs) override;
  virtual size_t get_input_batch() override;
  virtual int getInputWidth() const override;
  virtual int getInputHeight() const override;

 private:
  void pre_process(const std::vector<cv::Mat>& input_images,
                   ThreadSafeQueue<DpuInferenceTask>& task_queue);
  void dpu_inference(ThreadSafeQueue<DpuInferenceTask>& task_queue,
                     ThreadSafeQueue<DpuInferenceResult>& result_queue);
  void post_process(ThreadSafeQueue<DpuInferenceResult>& result_queue);

  SuperPointResult process_result(const DpuInferenceResult& result);

 private:
  std::vector<std::unique_ptr<vitis::ai::DpuTask>> runners_;
  std::vector<SuperPointResult> results_;
  std::vector<vitis::ai::library::InputTensor> input_tensors_;
  vector<size_t> chans_;

  int sWidth;
  int sHeight;
  size_t batch_;

  size_t channel1;
  size_t channel2;
  size_t outputH;
  size_t outputW;
  size_t output2H;
  size_t output2W;
  float conf_thresh;
  size_t outputSize1;
  size_t outputSize2;
};

SuperPoint::SuperPoint(const std::string& model_name, int num_runners) {}

SuperPoint::~SuperPoint() {}

std::unique_ptr<SuperPoint> SuperPoint::create(const std::string& model_name, int num_runners) {
  return std::unique_ptr<SuperPointImp>(new SuperPointImp(model_name, num_runners));
}

SuperPointImp::SuperPointImp(const std::string& model_name, int num_runners)
    : SuperPoint(model_name, num_runners) {
  for (int i = 0; i < num_runners; ++i) {
    runners_.emplace_back(vitis::ai::DpuTask::create(model_name));
  }
  input_tensors_ = runners_[0]->getInputTensor(0u);
  sWidth = input_tensors_[0].width;
  sHeight = input_tensors_[0].height;
  batch_ = input_tensors_[0].batch;
  chans_ = {65,256};
  auto output_tensors = sort_tensors(runners_[0]->getOutputTensor(0u), chans_);
  channel1 = output_tensors[0].channel;
  channel2 = output_tensors[1].channel;
  outputH = output_tensors[0].height;
  outputW = output_tensors[0].width;
  output2H = output_tensors[1].height;
  output2W = output_tensors[1].width;
  conf_thresh = 0.015;

  LOG_IF(INFO, ENV_PARAM(DEBUG_SUPERPOINT)) 
    << "tensor1 info : " << output_tensors[0].height << " " << output_tensors[0].width  << " " << output_tensors[0].channel << endl
    << "tensor2 info : " << output_tensors[1].height << " " << output_tensors[1].width  << " " << output_tensors[1].channel << endl;

  outputSize1 = output_tensors[0].channel * output_tensors[0].height * output_tensors[0].width;
  outputSize2 = output_tensors[1].channel * output_tensors[1].height * output_tensors[1].width;
}

SuperPointImp::~SuperPointImp() {}

size_t SuperPointImp::get_input_batch() { return runners_[0]->get_input_batch(0, 0); }
int SuperPointImp::getInputWidth() const {
  return runners_[0]->getInputTensor(0u)[0].width;
}
int SuperPointImp::getInputHeight() const {
  return runners_[0]->getInputTensor(0u)[0].height;
}

// L2_normalization function remains the same
inline void L2_normalization(const int8_t* input, float scale, int channel, int group, float* output) {
  for (int i = 0; i < group; ++i) {
    float sum = 0.0;
    for (int j = 0; j < channel; ++j) {
      int pos = i * channel + j;
      float temp = input[pos] * scale;
      sum += temp * temp;
    }
    float var = sqrt(sum);
    for (int j = 0; j < channel; ++j) {
      int pos = i * channel + j;
      output[pos] = (input[pos] * scale) / var;
    }
  }
}

// nms_mask and nms_fast functions remain the same
void nms_mask(vector<vector<int>>& grid, int x, int y, int dist_thresh) {
  int h = grid.size();
  int w = grid[0].size();
  for (int i = max(0, x - dist_thresh); i < min(h, x + dist_thresh + 1); ++i) {
    for (int j = max(0, y - dist_thresh); j < min(w, y + dist_thresh + 1); ++j) {
      grid[i][j] = -1;
    }
  }
  grid[x][y] = 1;
}

void nms_fast(const vector<int>& xs, const vector<int>& ys, const vector<float>& ptscore,
              vector<size_t>& keep_inds, const int inputW, const int inputH) {
  vector<vector<int>> grid(inputW, vector<int>(inputH, 0));
  vector<pair<float, size_t>> order;
  int dist_thresh = 4;
  for (size_t i = 0; i < ptscore.size(); ++i) {
    order.push_back({ptscore[i], i});
  }
  std::stable_sort(order.begin(), order.end(),
                   [](const pair<float, size_t>& ls, const pair<float, size_t>& rs) {
                     return ls.first > rs.first;
                   });
  vector<size_t> ordered;
  transform(order.begin(), order.end(), back_inserter(ordered),
            [](auto& km) { return km.second; });

  for (size_t _i = 0; _i < ordered.size(); ++_i) {
    size_t i = ordered[_i];
    int x = xs[i];
    int y = ys[i];
    if (grid[x][y] == 0 && x >= dist_thresh && x < inputW - dist_thresh && y >= dist_thresh &&
        y < inputH - dist_thresh) {
      keep_inds.push_back(i);
      nms_mask(grid, x, y, dist_thresh);
    }
  }
}

// bilinear_interpolation and grid_sample functions remain the same
float bilinear_interpolation(float v_xmin_ymin, float v_ymin_xmax, float v_ymax_xmin,
                             float v_xmax_ymax, int xmin, int ymin, int xmax, int ymax, float x,
                             float y, bool cout_value) {
  float value = v_xmin_ymin * (xmax - x) * (ymax - y) +
                v_ymin_xmax * (ymax - y) * (x - xmin) +
                v_ymax_xmin * (y - ymin) * (xmax - x) +
                v_xmax_ymax * (x - xmin) * (y - ymin);
  return value;
}

vector<vector<float>> grid_sample(const float* desc_map, const vector<pair<float, float>>& coarse_pts,
                                  const size_t channel, const size_t outputH, const size_t outputW) {
  vector<vector<float>> desc(coarse_pts.size());
  for (size_t i = 0; i < coarse_pts.size(); ++i) {
    float x = (coarse_pts[i].first + 1) / 8 - 0.5;
    float y = (coarse_pts[i].second + 1) / 8 - 0.5;
    int xmin = floor(x);
    int ymin = floor(y);
    int xmax = xmin + 1;
    int ymax = ymin + 1;
    // Bilinear interpolation
    {
      float divisor = 0.0;
      for (size_t j = 0; j < channel; ++j) {
        float value = bilinear_interpolation(
            desc_map[j + (ymin * outputW + xmin) * channel],
            desc_map[j + (ymin * outputW + xmax) * channel],
            desc_map[j + (ymax * outputW + xmin) * channel],
            desc_map[j + (ymax * outputW + xmax) * channel], xmin, ymin, xmax, ymax, x, y, false);
        divisor += value * value;
        desc[i].push_back(value);
      }
      for (size_t j = 0; j < channel; ++j) {
        desc[i][j] /= sqrt(divisor);  // L2 normalize
      }
    }
  }
  return desc;
}

// Pre-processing thread function
void SuperPointImp::pre_process(const std::vector<cv::Mat>& input_images,
                                ThreadSafeQueue<DpuInferenceTask>& task_queue) {
  for (size_t i = 0; i < input_images.size(); ++i) {
    cv::Mat img = input_images[i];
    DpuInferenceTask task;
    task.index = i;

    // Resize image
    cv::Mat resized_img;
    if (img.rows == sHeight && img.cols == sWidth) {
      resized_img = img;
    } else {
      cv::resize(img, resized_img, cv::Size(sWidth, sHeight));
    }
    task.scale_w = img.cols / (float)sWidth;
    task.scale_h = img.rows / (float)sHeight;

    // Normalize and scale
    cv::Mat float_img;
    cv::cvtColor(resized_img, float_img, cv::COLOR_BGR2GRAY);
    float_img.convertTo(float_img, CV_32FC1, 1.0 / 255.0);

    // Convert to int8
    cv::Mat input_img;
    float_img.convertTo(input_img, CV_8SC1, 1.0);

    // Copy data to input_data vector
    task.input_data.assign((int8_t*)input_img.data, (int8_t*)input_img.data + input_img.total());

    // Enqueue task
    task_queue.enqueue(task);
  }
  task_queue.shutdown();
}

// DPU inference thread function
void SuperPointImp::dpu_inference(ThreadSafeQueue<DpuInferenceTask>& task_queue,
                                  ThreadSafeQueue<DpuInferenceResult>& result_queue) {
  size_t runner_index = 0;
  size_t num_runners = runners_.size();
  std::vector<std::future<void>> futures;

  while (true) {
    DpuInferenceTask task;
    if (!task_queue.dequeue(task)) {
      break;
    }

    auto runner = runners_[runner_index % num_runners].get();
    runner_index++;

    // Prepare input tensor
    auto input_tensors = runner->getInputTensor(0u);
    int8_t* input_data = (int8_t*)input_tensors[0].get_data(0);
    memcpy(input_data, task.input_data.data(), task.input_data.size());

    // Run DPU inference asynchronously
    futures.emplace_back(std::async(std::launch::async, [this, runner, task, &result_queue]() {
      runner->run(0u);

      // Collect output tensors
      auto output_tensors = sort_tensors(runner->getOutputTensor(0u), chans_);

      DpuInferenceResult result;
      result.index = task.index;
      result.scale_w = task.scale_w;
      result.scale_h = task.scale_h;

      // Copy output data
      int8_t* out1 = (int8_t*)output_tensors[0].get_data(0);
      int8_t* out2 = (int8_t*)output_tensors[1].get_data(0);

      size_t size1 = output_tensors[0].size / output_tensors[0].batch;
      size_t size2 = output_tensors[1].size / output_tensors[1].batch;

      result.output_data1.assign(out1, out1 + size1);
      result.output_data2.assign(out2, out2 + size2);

      // Get scales
      result.scale1 = vitis::ai::library::tensor_scale(output_tensors[0]);
      result.scale2 = vitis::ai::library::tensor_scale(output_tensors[1]);

      // Enqueue result
      result_queue.enqueue(result);
    }));
  }

  // Wait for all inferences to complete
  for (auto& fut : futures) {
    fut.get();
  }
  result_queue.shutdown();
}

// Post-processing thread function
void SuperPointImp::post_process(ThreadSafeQueue<DpuInferenceResult>& result_queue) {
  while (true) {
    DpuInferenceResult result;
    if (!result_queue.dequeue(result)) {
      break;
    }

    // Process result
    SuperPointResult sp_result = process_result(result);

    // Store result
    results_[result.index] = sp_result;
  }
}

// Function to process a single result
SuperPointResult SuperPointImp::process_result(const DpuInferenceResult& result) {
  SuperPointResult sp_result;
  sp_result.index = result.index;
  sp_result.scale_w = result.scale_w;
  sp_result.scale_h = result.scale_h;

  // Post-processing steps
  const int8_t* out1 = result.output_data1.data();
  const int8_t* out2 = result.output_data2.data();

  float scale1 = result.scale1;
  float scale2 = result.scale2;

  vector<float> output1(outputSize1);

  // Softmax
#ifndef HW_SOFTMAX
  for (int i = 0; i < outputH * outputW; ++i) {
    float sum{0.0f};
    int pos = i * channel1;
    for (int j = 0; j < channel1; ++j) {
      output1[pos + j] = std::exp(out1[j + pos] * scale1);
      sum += output1[pos + j];
    }
    for (int j = 0; j < channel1; ++j) {
      output1[pos + j] /= sum;
    }
  }
#else
  vitis::ai::softmax(out1, scale1, channel1, outputH * outputW, output1.data());
#endif

  // Heatmap processing
  int reduced_size = (channel1 - 1) * outputH * outputW;
  vector<float> heatmap(reduced_size);
  // Remove heatmap[-1,:,:]
  for (size_t i = 0; i < outputH * outputW; i++) {
    memcpy(heatmap.data() + i * (channel1 - 1), output1.data() + i * channel1,
           sizeof(float) * (channel1 - 1));
  }

  // Keypoint detection
  vector<float> tmp;
  tmp.reserve(reduced_size);
  vector<int> xs, ys;
  vector<size_t> keep_inds;
  vector<float> ptscore;
  for (size_t m = 0u; m < outputH; ++m) {
    for (size_t i = 0u; i < 8; ++i) {
      for (size_t n = 0u; n < outputW; ++n) {
        for (size_t j = 0u; j < 8; ++j) {
          tmp.push_back(heatmap.at(i * 8 + j + (m * outputW + n) * 64));  // transpose heatmap
          if (tmp.back() > conf_thresh) {
            ys.push_back(m * 8 + i);
            xs.push_back(n * 8 + j);
            ptscore.push_back(tmp.back());
          }
        }
      }
    }
  }

  // NMS
  nms_fast(xs, ys, ptscore, keep_inds, sWidth, sHeight);

  // L2 Normalization
  vector<float> output2(outputSize2);
  L2_normalization(out2, scale2, channel2, output2H * output2W, output2.data());

  // Descriptor extraction
  for (size_t i = 0; i < keep_inds.size(); ++i) {
    pair<float, float> pt(float(xs[keep_inds[i]]), float(ys[keep_inds[i]]));
    sp_result.keypoints.push_back(pt);
  }

  sp_result.descriptor = grid_sample(output2.data(), sp_result.keypoints, channel2, output2H, output2W);

  return sp_result;
}

// Run function
std::vector<SuperPointResult> SuperPointImp::run(const std::vector<cv::Mat>& imgs) {
  results_.resize(imgs.size());

  ThreadSafeQueue<DpuInferenceTask> task_queue;
  ThreadSafeQueue<DpuInferenceResult> result_queue;

  // Start pre-processing thread
  std::thread preproc_thread(&SuperPointImp::pre_process, this, std::ref(imgs), std::ref(task_queue));

  // Start DPU inference thread
  std::thread dpu_thread(&SuperPointImp::dpu_inference, this, std::ref(task_queue), std::ref(result_queue));

  // Start post-processing thread
  std::thread postproc_thread(&SuperPointImp::post_process, this, std::ref(result_queue));

  // Wait for threads to finish
  preproc_thread.join();
  dpu_thread.join();
  postproc_thread.join();

  return results_;
}

}  // namespace ai
}  // namespace vitis
