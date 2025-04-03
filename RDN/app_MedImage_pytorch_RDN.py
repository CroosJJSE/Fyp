#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: MIT

# Author: Doxxxx
# date:June 2024
# This program provide a sample of performing the super-resolution operation.

print(" ")
print("RDN Network (pre-trained with Kvasir Dataset) and quantize in Kvasir data subset in pytorch")
print(" ")

# ***********************************************************************
# Import Packages
# ***********************************************************************
import os
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

# ***********************************************************************
# input file names, include the model and test image path.
# ***********************************************************************
cnn_xmodel  = os.path.join("./RDN_pt", "RDN_pt.xmodel")
#labels_file = os.path.join("./"        , "cifar10_labels.dat")
hrimages_dir  = os.path.join("./"        , "medicalx2df", "train", "HR_x2")
lrimages_dir  = os.path.join("./"        , "medicalx2df", "train", "LR_x2")

# ***********************************************************************
# utility Functions
# ***********************************************************************
def plt_imshow(title, image):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.savefig(title)
    plt.show()

def predict_label(softmax):
    with open(labels_file, "r") as f:
        lines = f.readlines()
    return lines[np.argmax(softmax)]

# see https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def calculate_softmax(data):
    result = np.exp(data)
    return result

# ***********************************************************************
# Pre-Processing Functions for Dataset
# ***********************************************************************

#Normalize the input images
def Normalize(image):
    x_test  = np.asarray(image)
    x_test = x_test.astype(np.float32)
    x_test = x_test/255.0
    out_x_test = x_test
    return out_x_test

#preprocess the input images, the input images' channels are BGR channels.
def preprocess_fn(image_filename):
    #read the source image
    image=cv2.imread(image_filename)
    #print("Image width is:{:2d}, Image height is:{:2d}".format(image.shape[1], image.shape[0]))
    
    #these codes save the source image
    '''
    FileName = image_filename.split('.', -1)[-2]
    SaveFileName = FileName.split('/',-1)[4]+'Src.bmp'
    print("FileName is:{:s}".format(SaveFileName))
    #image2 = imageY.astype(np.float32)
    cv2.imwrite(SaveFileName, image)
    '''
    
    #Normalize the source images
    image2 = Normalize(image) #added by me for ResNet18

    return image2

# ***********************************************************************
# Prepare the Overlay and load the "RDN_pt.xmodel"
# ***********************************************************************
from pynq_dpu import DpuOverlay
overlay = DpuOverlay("dpu.bit")
overlay.load_model(cnn_xmodel)

# ***********************************************************************
# Use VART APIs
# ***********************************************************************

#load the filename of high and low resolution image
hr_original_images = [i for i in os.listdir(hrimages_dir) if i.endswith("png")]
lr_original_images = [i for i in os.listdir(lrimages_dir) if i.endswith("png")]
#get the image number
total_images = len(hr_original_images)
print("Total image is:{:2d}".format(total_images))

#acquire network structure
dpu = overlay.runner
inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()
shapeIn = tuple(inputTensors[0].dims)
shapeOut = tuple(outputTensors[0].dims)
outputSize = int(outputTensors[0].get_data_size() / shapeIn[0])
print("shapeIn   : {}".format(shapeIn))
print("shapeOut  : {}".format(shapeOut))
print("outputSize: {}".format(outputSize))

# allocate some buffers that will be re-used multiple times
predictions = np.empty(total_images)
test_labels = np.empty(total_images)
softmax = np.empty(outputSize)
output_data = [np.empty(shapeOut, dtype=np.float32, order="C")]
input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
image = input_data[0]

# ***********************************************************************
# Run DPU to Make Predictions on ALL the images
# ***********************************************************************
print("Classifying {} CIFAR10 pictures ...".format(total_images))
time1 = time.time()
for image_index in range(total_images):
    #input image filenames
    hrfilename = os.path.join(hrimages_dir, hr_original_images[image_index])
    lrfilename = os.path.join(lrimages_dir, lr_original_images[image_index])

    #preprocess the input images
    preprocessed = preprocess_fn(lrfilename)

    #convert input image to model input
    image[0,...] = preprocessed.reshape(shapeIn[1:])

    #run super-resolution model
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)

    #acquire model output
    temp = [j.reshape(1, outputSize) for j in output_data]
    
    #convert the model output to result image
    result = temp[0][0].reshape(shapeOut[1:])
    result = (result)*255.0
    result = np.clip(result, 0.0, 255.0)
    
    print(result.shape)
    print("Image width is:{:d}, Image height is:{:d}".format(result.shape[1], result.shape[0]))
    print("Center Gray is:{:f}".format(result[512, 640, 0]))
    resultOut = result.astype(np.uint8)

    #save the output image
    
    strSaveName =hr_original_images[image_index].split('.', -1)[-2] + "result.bmp"
    cv2.imwrite(strSaveName, resultOut)
    print("Saveimage name is:{:s}".format(hr_original_images[image_index].split('.', -1)[-2]))
    

time2 = time.time()
execution_time = time2-time1
print("  Execution time: {:.4f}s".format(execution_time))
print("      Throughput: {:.4f}FPS".format(total_images/execution_time))

# ***********************************************************************
# Clean up
# ***********************************************************************
del overlay
del dpu
