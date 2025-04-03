#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include "./superpoint.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
  cout << "Starting SuperPoint execution." << endl;

  if(argc < 3) {
    cout << "Warning: Usage: " << argv[0] << " model_name image" << endl;
    char* default_argv[] = {(char*)"demo", (char*)"superpoint_tf.xmodel", (char*)"test.jpg"};
    argv = default_argv;
  }

  int nIter = 1;
  string model_name = argv[1];
  cout << "Model name: " << model_name << endl;

  cout << "Loading image: " << argv[2] << endl;
  Mat img = imread(argv[2], cv::IMREAD_COLOR);  // Ensure 3-channel image
  if (img.empty()) {
    cerr << "Error: Failed to load image: " << argv[2] << endl;
    return -1;
  }
  cout << "Image loaded successfully." << endl;

  cout << "Creating SuperPoint model instance." << endl;
  auto superpoint = vitis::ai::SuperPoint::create(model_name, 5);
  if (!superpoint) {
    cerr << "Error: Failed to create SuperPoint instance." << endl;
    abort();
  }
  cout << "SuperPoint model created successfully." << endl;

  vector<Mat> imgs;
  int batch_size = superpoint->get_input_batch();
  cout << "Input batch size: " << batch_size << endl;

  for(size_t i = 0; i < batch_size; ++i)
    imgs.push_back(img);

  cout << "Running SuperPoint model." << endl;
  auto start = chrono::high_resolution_clock::now();
  auto result = superpoint->run(imgs);

  for(int i = 1; i < nIter; ++i) {
    result = superpoint->run(imgs);
  }

  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>((end - start));

  for(size_t i = 0; i < batch_size; ++i) {
    cout << "Processing batch " << i + 1 << " / " << batch_size << endl;
    cout << "Result scales: " << result[i].scale_h << " " << result[i].scale_w << endl;
    for(size_t k = 0; k < result[i].keypoints.size(); ++k)
      circle(imgs[i], Point(result[i].keypoints[k].first * result[i].scale_w,
             result[i].keypoints[k].second * result[i].scale_h), 1, Scalar(0, 0, 255), -1);

    string output_filename = "result_superpoint_" + to_string(i) + ".jpg";
    imwrite(output_filename, imgs[i]);
    cout << "Saved result image: " << output_filename << endl;
  }

  cout << "Processing complete. Average Time: " << duration.count() / nIter << " ms" << endl;
  cout << "BYEBYE" << endl;
  return 0;
}
