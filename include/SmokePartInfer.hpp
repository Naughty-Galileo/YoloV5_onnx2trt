#ifndef SMOKE_PART_INFER_H
#define SMOKE_PART_INFER_H

#include <memory>
#include <map>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include "opencv2/core.hpp"

using namespace nvonnxparser;
using namespace cv;
using namespace std;
using namespace nvinfer1;

class SmokePartInfer
{
    public:
        typedef struct {
            Rect2d m_cRect;
            float m_fScore;
            int m_eType;
            vector<float> Dpoints;
        } DetItem;
        
        SmokePartInfer();
        ~SmokePartInfer();
        
        bool runDetect(const cv::Mat& cInMat, const float Thresh, const int smokePartAreas, std::vector<DetItem>& smokeDets, int xmin, int ymin);
        bool doNms(std::vector<DetItem>& smokeDets, const int smokePartAreas);
        
        bool doPreprocess(const cv::Mat& cInMat);
        bool doInference();
        bool filterByThresh(const float Thresh, std::vector<DetItem>& smokeDets);
        bool recoverPosInfo(std::vector<DetItem>& smokeDets, int xmin, int ymin);

    private:    
        nvinfer1::ICudaEngine *smoke_m_CudaEngine; 
        // nvinfer1::IRuntime *smoke_m_CudaRuntime;
        nvinfer1::IExecutionContext *smoke_m_CudaContext;
        cudaStream_t smoke_m_CudaStream;
        
        void* smoke_m_ArrayDevMemory[2]{0};
        void* smoke_m_ArrayHostMemory[2]{0};
        int smoke_m_ArraySize[2]{0};
        int smoke_m_iInIndex;
        int smoke_m_iOutIndex;
    private:
        int smoke_m_iInHeight;
        int smoke_m_iInWidth;
        cv::Size smoke_m_cModelInputSize;
        cv::Mat smoke_m_cRGBMat;
        cv::Mat smoke_m_Normalized;
        cv::Mat smoke_m_cPasteBoard;
        std::vector<cv::Mat> smoke_m_InputWrappers{};

        int smoke_m_iPadDeltaX = 0;
        int smoke_m_iPadDeltaY = 0;
        float smoke_m_fRecoverScale = 1.f;
};
#endif 
