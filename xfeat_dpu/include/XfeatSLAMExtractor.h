

#ifndef XFEXTRACTOR_H
#define XFEXTRACTOR_H
#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
// #include "torch/torch.h"


namespace ORB_SLAM3
{

class XSLAMFextractor
{
public:
    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;   
    XSLAMFextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST){
                    mvScaleFactor.resize(nlevels);
                    mvLevelSigma2.resize(nlevels);
                    mvScaleFactor[0]=1.0f;
                    mvLevelSigma2[0]=1.0f;
                    for(int i=1; i<nlevels; i++)
                    {
                        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
                        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
                    }
            
                    mvInvScaleFactor.resize(nlevels);
                    mvInvLevelSigma2.resize(nlevels);
                    for(int i=0; i<nlevels; i++)
                    {
                        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
                        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
                    }
                    
                    mvImagePyramid.resize(nlevels);
            
                    mnFeaturesPerLevel.resize(nlevels);
                    float factor = 1.0f / scaleFactor;
                    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));
            
                    int sumFeatures = 0;
                    for( int level = 0; level < nlevels-1; level++ )
                    {
                        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
                        sumFeatures += mnFeaturesPerLevel[level];
                        nDesiredFeaturesPerScale *= factor;
                    }
                    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);
            
                    //This is for orientation
                    // pre-compute the end of a row in a circular patch
                    umax.resize(HALF_PATCH_SIZE + 1);
            
                    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
                    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
                    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
                    for (v = 0; v <= vmax; ++v)
                        umax[v] = cvRound(sqrt(hp2 - v * v));
            
                    // Make sure we are symmetric
                    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
                    {
                        while (umax[v0] == umax[v0 + 1])
                            ++v0;
                        umax[v] = v0;
                        ++v0;
                    }
            
                    // load the interpolators
                    // bilinear = std::make_shared<InterpolateSparse2d>("bilinear");     
                    // nearest  = std::make_shared<InterpolateSparse2d>("nearest"); 
                }

    ~XSLAMFextractor(){}

    virtual int operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea) = 0;

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:
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
};

} //namespace ORB_SLAM

#endif
