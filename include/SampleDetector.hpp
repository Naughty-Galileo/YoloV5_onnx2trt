#ifndef COMMON_DET_INFER_H
#define COMMON_DET_INFER_H

#include <map>
#include <cmath>
#include <memory>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <NvInfer.h>
#include <glog/logging.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include "logging.h"

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace cv;
using namespace std;

class SampleDetector {

public:
    typedef struct {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        int label;
    } DetItem;

    SampleDetector();
    ~SampleDetector();
    
    // 初始化检测器资源，模型加载等操作
    bool Init(const std::string& strModelName); 
    // 去初始化检测器资源 
    bool UnInit(); 

    // 业务处理函数，输入分析图片，返回算法分析结果
    bool ProcessImage(const cv::Mat& img, std::vector<DetItem>& DetObj);

private:
    //预处理
    bool doPreprocess(const cv::Mat& cInMat);
    //推理
    bool doInference();
    
    // nms前 根据mConf_thresh进行过滤
    bool filter(std::vector<DetItem>& vecDets);

    // nms操作
    bool doNms(std::vector<DetItem>& vecDets);

    // 将坐标映射会原图
    bool recoverPosInfo(std::vector<DetItem>& vecDets);

private:
    size_t mClasses = 0;
    int m_iBoxNums = 0;
    
    double mIou_thresh = 0.3;
    double mConf_thresh = 0.5;
    
    
    nvinfer1::ICudaEngine *m_CudaEngine; 
    nvinfer1::IExecutionContext *m_CudaContext;
    cudaStream_t m_CudaStream;
    
    int m_iInIndex;
    int m_iOutIndex;
    
    cv::Size m_cModelInputSize;
    cv::Mat m_cRGBMat;
    cv::Mat m_Normalized;
    cv::Mat m_cPasteBoard;
    std::vector<cv::Mat> m_InputWrappers{};

    void* m_ArrayDevMemory[2]{0};
    void* m_ArrayHostMemory[2]{0};
    int m_ArraySize[2]{0};
    
    int m_iInHeight;
    int m_iInWidth;
    int m_iPadDeltaX = 0;
    int m_iPadDeltaY = 0;
    float m_fRecoverScale = 1.f; 
    
    bool m_bUninit = false;
    
    int img_w = 0;
    int img_h = 0;
};

#endif //JI_SAMPLEDETECTOR_HPP
