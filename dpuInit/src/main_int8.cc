//# SPDX-License-Identifier: MIT

// based on  Vitis AI 1.4 demo code
// Author: Hui Shen(Doxxxx)
// Date: June, 2024

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "common.h"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

GraphInfo shapes;

string baseImagePath, wordsPath;  // they will get their values via argv[]

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const& path, vector<string>& hrimages, vector<string>& lrimages) {
  hrimages.clear();
  lrimages.clear();
  struct dirent* entry;


  string hrpath = path + string("HR_x2"); 
  string lrpath = path + string("LR_x2");
  
  /*Check if path is a valid directory path. */
  struct stat hrs, lrs;
  lstat(hrpath.c_str(), &hrs);
  lstat(lrpath.c_str(), &lrs);


  if (!S_ISDIR(hrs.st_mode)||!S_ISDIR(lrs.st_mode)) {
    fprintf(stderr, "Error: %s or %s is not a valid directory!\n", hrpath.c_str(), lrpath.c_str());
    exit(1);
  }

  DIR* hrdir = opendir(hrpath.c_str());
  if (hrdir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", hrpath.c_str());
    exit(1);
  }


  DIR* lrdir = opendir(lrpath.c_str());
  if (lrdir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", lrpath.c_str());
    exit(1);
  }

  while ((entry = readdir(hrdir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
           hrimages.push_back(name);
      }
    }
  }

  while ((entry = readdir(lrdir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
           lrimages.push_back(name);
      }
    }
  }


  closedir(hrdir);
  closedir(lrdir);
}


/**
 * @brief Get PSNR between the processed and input image
 *
 * @param matProcessedImage Processed by CNN in DPU
 * @param matDstImage the compared image
 *
 * @return figured PSNR
 */

double FigurePSNR(Mat matProcessedImage, Mat matDstImage){
   uint8_t* pmatProcessedImage = matProcessedImage.ptr<uint8_t>(0);
   uchar* pmatDstImage = matDstImage.ptr<uchar>(0);
   cout << "Line " << __LINE__ <<endl;
   //cout << "Processed image width is:" << matProcessedImage.cols << ",Processed image height is:" << matProcessedImage.rows << endl;
   //cout << "Destination image width is:" << matDstImage.cols << ", Destination image height is:" << matDstImage.rows <<endl;


   int n, m;
   int nSize = matProcessedImage.rows* matProcessedImage.cols;
   double dSum  =0.0;
   for(n = 0; n < nSize; n++){
      for(m = 0; m < 3; m++){
        dSum += (static_cast<double>(pmatProcessedImage[n]) - pmatDstImage[n]) * (static_cast<double>(pmatProcessedImage[n]) - pmatDstImage[n]);
      }	   
   }
   cout << "Line " << __LINE__ << endl;
   dSum = dSum/nSize/3.0;
   double dPSNR = 10*log10(255.*255./dSum);
   
   return dPSNR;
}

/**
 * @brief Cut off the float number to unsigned char
 *
 * @param fInputV input value
 *
 * @return result value
 */

uchar CutoffValue(float fInputV){
   if(fInputV > 255.0f){
      return 255;
   }else if(fInputV < 0.0f){
      return 0;
   }else{
      return static_cast<uchar>(fInputV + 0.5f);
   }
}


/**
 * @brief Convert processed image to result image
 *
 * @param matSrcImage the image output by the DPU net
 * @param matDstImage the result image
 *
 * @return None
 */

void convertImage(cv::Mat& matSrcImage, cv::Mat& matProcessedImage, float fScale){
     cv::Vec3f* pscSrcImage = matSrcImage.ptr<cv::Vec3f>(0);
     cv::Vec3b* pucProcessedImage = matProcessedImage.ptr<cv::Vec3b>(0);

     int nSize = matSrcImage.rows * matSrcImage.cols;

     for(int n=0; n< nSize; n++){
	for(int m = 0; m < 3; m++){
	   pucProcessedImage[n][m] = CutoffValue((pscSrcImage[n][2-m])*255.0f);
	}     
     }

}


/**
 * @brief Run DPU Task for CNN
 *
 * @return none
 */
void run_CNN(vart::Runner* runner) {

  vector<string> kinds, hrimages, lrimages;

  /* Load all image names.*/
  ListImages(baseImagePath, hrimages, lrimages);
  if (hrimages.size() == 0 || lrimages.size() == 0) {
    cerr << "\nError: No images existing under " << baseImagePath << endl;
    return;
  }

  /* get in/out tensors and dims*/
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_shape(); //_dims();
  auto in_dims = inputTensors[0]->get_shape(); //dims();

  auto input_scale = get_input_scale(inputTensors[0]);
  auto output_scale = get_output_scale(outputTensors[0]);

  /*get shape info*/
  int outSize = shapes.outTensorList[0].size;
  int inSize = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int outHeight = shapes.outTensorList[0].height;
  int outWidth = shapes.outTensorList[0].width;
  int batchSize = in_dims[0];

  //for debug
  //cout << "OUT  dims " << out_dims  << endl;
  cout << "OUT  size " << outSize   << endl;
  //cout << "IN   dims " << in_dims   << endl;
  cout << "IN   size " << inSize    << endl;
  cout << "IN Height " << inHeight  << endl;
  cout << "IN Width  " << inWidth   << endl;
  cout << "batchSize " << batchSize << endl;
  cout << "input scale is:" << input_scale << endl;
  cout << "output scale is:" << output_scale << endl;


  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;

  //vector<Mat> imageList;
  float* imageInputs = new float[inSize * batchSize];

  float* softmax = new float[outSize];
  float* FCResult = new float[batchSize * outSize];
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;
  std::vector<Mat>vmatYCrCb;

  double dAveragePSNR = 0.0;
  /*run with batch*/
  for (unsigned int n = 0; n < hrimages.size(); n += batchSize) {
    unsigned int runSize =
        (hrimages.size() < (n + batchSize)) ? (hrimages.size() - n) : batchSize;
    in_dims[0] = runSize;
    out_dims[0] = batchSize;
    cout << "runSize is:" << runSize <<endl;
    cout << "batchSize is:" << batchSize <<endl;
    for (unsigned int i = 0; i < runSize; i++) {
      string strLRName = baseImagePath + "LR_x2/";
      strLRName = strLRName + lrimages[n + i];      
      Mat lrimage = imread(strLRName);
      cout << "File name is:" << strLRName<<endl;
      cout << "lrimage Height is:" << lrimage.rows << ", lrimage Width is:" << lrimage.cols <<endl;
      /*image pre-process*/
      size_t szSrcPos = lrimages[n+i].find('.'); 
      string strSrcName = lrimages[n + i].substr(0 ,szSrcPos);
      cout << "Src Name:" << strSrcName << endl;
      
      /*Save the source images*/
      char cFileName[100];
      sprintf(cFileName, "Src%s%d%d.bmp",strSrcName.c_str(), n, i);
      imwrite(cFileName, lrimage);

      //matY.convertTo(image2, CV_32SC1);
      for (int h = 0; h < inHeight; h++) {
	for (int w = 0; w < inWidth; w++) {
	  for (int c = 0; c < 3; c++) {
	    imageInputs[i * inSize+h*inWidth*3+w*3 +2-c] = lrimage.at<Vec3b>(h, w)[c]/255.0f;
          }
        }
      }
      
    }

    /* in/out tensor refactory for batch inout/output */
    //batchTensors.push_back(std::shared_ptr<xir::Tensor>(
    //    xir::Tensor::create(inputTensors[0]->get_name(), in_dims,
    //                        xir::DataType{xir::DataType::XINT, 8u})));
        /* in/out tensor refactory for batch inout/output */
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(inputTensors[0]->get_name(), in_dims,
                            xir::DataType{xir::DataType::FLOAT, 32})));

                            
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        imageInputs, batchTensors.back().get()));
    //batchTensors.push_back(std::shared_ptr<xir::Tensor>(
    //    xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
    //                        xir::DataType{xir::DataType::XINT, 8u})));
    
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
                            xir::DataType{xir::DataType::FLOAT, 32})));

                            
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        FCResult, batchTensors.back().get()));
    cout << "LINE " << __LINE__ << endl;
    /*tensor buffer input/output */
    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    cout << "LINE " << __LINE__ << endl;
    /*run*/
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);
    cout << "LINE "<< __LINE__ << endl;

    for (unsigned int i = 0; i < runSize; i++) {
      /* Calculate the PSNR on CPU*/
      string strHRImages = baseImagePath + "HR_x2/";
      strHRImages = strHRImages + hrimages[n+i];

      cout << "HRImage Name is:" << strHRImages <<endl;
      Mat hrimage = imread(strHRImages);


      cv::Mat matProcessedImage = cv::Mat(outHeight, outWidth, CV_32FC3, &(FCResult[i*outSize]));
      cv::Mat matResultImage = cv::Mat(outHeight, outWidth, CV_8UC3);

      convertImage(matProcessedImage, matResultImage, output_scale);
      double dPSNR = FigurePSNR(matResultImage, hrimage);
      
      dAveragePSNR += dPSNR;
      
      size_t stPos = hrimages[n+i].find('.');
      string strDstName =  hrimages[n+i].substr(0, stPos);
      cout << "Dst Name:" << strDstName << endl;
      
      /*save the super-resolution images*/
      char cFileName[100];
      sprintf(cFileName, "Dst%s%d%d.bmp", strDstName.c_str(),n,i);
      imwrite(cFileName, matResultImage);
      
      cout << "The PSNR of" << hrimages[n+i] << " is:" << dPSNR <<endl;
    }
    inputs.clear();
    outputs.clear();
  }

  dAveragePSNR = dAveragePSNR/(hrimages.size());
  cout << "The average PSNR is:" << dAveragePSNR <<endl;

  delete[] FCResult;
  delete[] imageInputs;
  delete[] softmax;
}

/**
 * @brief Entry for running CNN
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy CNN on DPU platform.
 *
 */
int main(int argc, char* argv[])
{
  cout << argc << std::endl;
  // Check args
  if (argc != 3) {
    cout << "Usage: run_cnn model_pathName test_images_pathname" << endl;
    return -1;
  }

  baseImagePath = std::string(argv[2]); //path name of the folder with test images
  //wordsPath     = std::string(argv[3]); //filename of the labels

  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "CNN should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();
  /*create runner*/
  auto runner = vart::Runner::create_runner(subgraph[0], "run");
  // ai::XdpuRunner* runner = new ai::XdpuRunner("./");
  /*get in/out tensor*/
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  /*get in/out tensor shape*/
  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

  /*run with batch*/
  run_CNN(runner.get());
  return 0;
}
