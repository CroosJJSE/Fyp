#ifndef XFEXTRACTOR_H
#define XFEXTRACTOR_H

#include "XFeat.h"
#include "XFextractorBase.h"
#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
//#include "torch/torch.h"


namespace ORB_SLAM3
{

class XFextractor : public XFextractorBase {
    public:
        XFextractor(int nfeatures, float scaleFactor, int nlevels,
                    int iniThFAST, int minThFAST);
    
        ~XFextractor() override = default;
    
        int operator()(cv::InputArray _image, cv::InputArray _mask,
                        std::vector<cv::KeyPoint>& _keypoints,
                        cv::OutputArray _descriptors, std::vector<int>& vLappingArea) override;
    
        int GetLevels() override { return nlevels; }
        float GetScaleFactor() override { return scaleFactor; }
        std::vector<float> GetScaleFactors() override { return mvScaleFactor; }
        std::vector<float> GetInverseScaleFactors() override { return mvInvScaleFactor; }
        std::vector<float> GetScaleSigmaSquares() override { return mvLevelSigma2; }
        std::vector<float> GetInverseScaleSigmaSquares() override { return mvInvLevelSigma2; }

        std::vector<cv::Mat> mvImagePyramid;

protected:
    std::string getModelWeightsPath(std::string weights);
    torch::Tensor parseInput(cv::Mat& img);
    std::tuple<torch::Tensor, double, double> preprocessTensor(torch::Tensor& x);
    torch::Tensor getKptsHeatmap(torch::Tensor& kpts, float softmax_temp=1.0);
    torch::Tensor NMS(torch::Tensor& x, float threshold = 0.05, int kernel_size = 5);

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    torch::DeviceType device_type;
    std::shared_ptr<XFeatModel> model;
    std::shared_ptr<InterpolateSparse2d> bilinear, nearest;
};

} //namespace ORB_SLAM

#endif

