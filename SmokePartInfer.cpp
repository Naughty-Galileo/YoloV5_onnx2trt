#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <glog/logging.h>

#include "Yolov5TrtInfer.hpp"

#include "NvOnnxParser.h"
#include "./logging.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "NvInfer.h"

#define IN_NAME_SMOKE "input"
#define OUT_NAME_SMOKE "output"

using namespace nvinfer1;
using namespace nvonnxparser;

float smoke_boxIOU(const cv::Rect2d &rect1, const cv::Rect2d &rect2)
{
    cv::Rect2d intersecRect =  rect1 & rect2;
    return intersecRect.area() / ( rect1.area() + rect2.area() - intersecRect.area() );
}

bool smoke_exists(const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

Yolov5TrtInfer::Yolov5TrtInfer()
{
    if(! smoke_exists("../model/smoke.trt"))
    {
        Logger gLogger;
        IRuntime* smoke_m_CudaRuntime = createInferRuntime(gLogger);
        IBuilder* smoke_builder = createInferBuilder(gLogger.getTRTLogger());
        cout<<"debug 1-3"<<endl;
        smoke_builder->setMaxBatchSize(1);
        cout<<"debug 2"<<endl;
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition* smoke_network = smoke_builder->createNetworkV2(explicitBatch);
        cout<<"debug 3"<<endl;
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*smoke_network, gLogger);
        parser->parseFromFile("../model/smoke-best.onnx", static_cast<int>(ILogger::Severity::kWARNING));
        cout<<"debug 4"<<endl;
        IBuilderConfig* config = smoke_builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1ULL << 30);
        cout<<"debug 5"<<endl;
        smoke_m_CudaEngine = smoke_builder->buildEngineWithConfig(*smoke_network, *config);
        smoke_m_CudaContext = smoke_m_CudaEngine->createExecutionContext();
        
        IHostMemory *gieModelStream = smoke_m_CudaEngine->serialize();
        std::string serialize_str;
        std::ofstream serialize_output_stream;
        serialize_str.resize(gieModelStream->size());   
        memcpy((void*)serialize_str.data(), gieModelStream->data(), gieModelStream->size());
        serialize_output_stream.open("../model/smoke.trt");
        serialize_output_stream<<serialize_str;
        serialize_output_stream.close();
        parser->destroy();
        smoke_network->destroy();
        config->destroy();
        smoke_builder->destroy();
    }
    else{
        Logger gLogger;
        IRuntime* smoke_m_CudaRuntime = createInferRuntime(gLogger);
        std::string cached_path = "../model/smoke.trt";
        std::ifstream fin(cached_path);
        std::string cached_engine = "";
        while (fin.peek() != EOF){ 
                std::stringstream buffer;
                buffer << fin.rdbuf();
                cached_engine.append(buffer.str());
        }
        fin.close();
        smoke_m_CudaEngine = smoke_m_CudaRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
        smoke_m_CudaContext = smoke_m_CudaEngine->createExecutionContext();
    }

    // 分配输入输出的空间,DEVICE侧和HOST侧
    smoke_m_iInIndex = smoke_m_CudaEngine->getBindingIndex( IN_NAME_SMOKE );
    smoke_m_iOutIndex = smoke_m_CudaEngine->getBindingIndex(OUT_NAME_SMOKE);

    Dims dims_i = smoke_m_CudaEngine->getBindingDimensions(smoke_m_iInIndex);
    cout << "input dims " << dims_i.d[0] << " " << dims_i.d[1] << " " << dims_i.d[2] << " " << dims_i.d[3] << endl;
    
    int size = dims_i.d[0] * dims_i.d[1] * dims_i.d[2] * dims_i.d[3];
    smoke_m_cModelInputSize = cv::Size(dims_i.d[3], dims_i.d[2]);
    
    cudaMalloc(&smoke_m_ArrayDevMemory[smoke_m_iInIndex], size * sizeof(float));
    smoke_m_ArrayHostMemory[smoke_m_iInIndex] = malloc(size * sizeof(float));
    
    //方便NHWC到NCHW的预处理
    smoke_m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, smoke_m_ArrayHostMemory[smoke_m_iInIndex]);
    smoke_m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, smoke_m_ArrayHostMemory[smoke_m_iInIndex] + sizeof(float) * dims_i.d[2] * dims_i.d[3] );
    smoke_m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, smoke_m_ArrayHostMemory[smoke_m_iInIndex] + 2 * sizeof(float) * dims_i.d[2] * dims_i.d[3]);
    smoke_m_ArraySize[smoke_m_iInIndex] = size *sizeof(float);

    dims_i = smoke_m_CudaEngine->getBindingDimensions(smoke_m_iOutIndex);
    size = dims_i.d[0] * dims_i.d[1] * dims_i.d[2];
    cout << "output dims "<< dims_i.d[0] << " " << dims_i.d[1] << " " << dims_i.d[2] << endl;
    cudaMalloc(&smoke_m_ArrayDevMemory[smoke_m_iOutIndex], size * sizeof(float));
    smoke_m_ArrayHostMemory[smoke_m_iOutIndex] = malloc( size * sizeof(float));
    smoke_m_ArraySize[smoke_m_iOutIndex] = size *sizeof(float);
    cudaStreamCreate(&smoke_m_CudaStream);
    
    smoke_m_cPasteBoard = cv::Mat(smoke_m_cModelInputSize, CV_8UC3, cv::Scalar(128, 128, 128));
}

Yolov5TrtInfer::~Yolov5TrtInfer()
{
    for(auto &p:smoke_m_ArrayDevMemory)
    {
        cudaFree(p);
        p = nullptr;
    }
    for(auto &p: smoke_m_ArrayHostMemory)
    {        
        free(p);
        p = nullptr;        
    } 
    
    smoke_m_CudaContext->destroy();
    smoke_m_CudaEngine->destroy();
    cudaStreamDestroy(smoke_m_CudaStream);
}

bool Yolov5TrtInfer::runDetect(const cv::Mat& cInMat,const float Thresh, const int smokePartAreas, std::vector<DetItem>& smokeDets, int xmin, int ymin)
{
    doPreprocess(cInMat);
    doInference();
    smokeDets.clear();       
    float* pData = (float*)smoke_m_ArrayHostMemory[smoke_m_iOutIndex];
    
    int iIndex = 0;
    float fConf = 0, x = 0, y = 0, w = 0, h = 0, r_x = 0, r_y = 0;
    int iBestClassId = 0;  

    for(int i=0;i<6300;++i)
    {
        iIndex = i * 10 + 4;
        fConf = pData[iIndex];
        if (fConf > 0.05)
        {
            h = pData[--iIndex];
            w = pData[--iIndex];
            y = pData[--iIndex];
            x = pData[--iIndex];

            iIndex += 5;

            std::vector<float> points;

            for(int j=0;j<2;j++)
            {
                points.emplace_back((float)pData[iIndex++]);
                points.emplace_back((float)pData[iIndex++]);
            }

            iBestClassId = iIndex;
            if (fConf*pData[iBestClassId] < 0.05) continue;         
            smokeDets.push_back( {cv::Rect2d(x-w/2, y-h/2, w, h), fConf*pData[iBestClassId], iBestClassId - iIndex, points});                 
        }
    }
    //阈值过滤
    filterByThresh(Thresh, smokeDets);
    //恢复位置到原图    
    recoverPosInfo(smokeDets,xmin,ymin);
    //nms
    doNms(smokeDets, smokePartAreas);
    return true;
}

bool Yolov5TrtInfer::doNms(std::vector<DetItem>& smokeDets, const int smokePartAreas)
{
    std::sort(smokeDets.begin(),smokeDets.end(),[](const DetItem &det1, const DetItem &det2){return det1.m_fScore > det2.m_fScore;});
    for (int i = 0; i < smokeDets.size(); ++i)
    {
        if (smokeDets[i].m_fScore < 0.00000000001f)
        {
            continue;
        }
        if(smokeDets[i].m_cRect.area()< smokePartAreas){
            smokeDets[i].m_fScore = 0;
            continue;
        }
        for (int j = i+1; j < smokeDets.size(); ++j)
        {
            if (smokeDets[i].m_eType == smokeDets[j].m_eType) 
            {
                if(smoke_boxIOU(smokeDets[i].m_cRect, smokeDets[j].m_cRect) >= 0.2)
                {
                    smokeDets[j].m_fScore = 0;
                }
            }
        }
    }

    for( auto iter = smokeDets.begin(); iter != smokeDets.end(); )
    {
        if( iter->m_fScore < 0.000000001f )
        {
            iter = smokeDets.erase(iter);
        }
        else
        {
            iter++;
        }
    }
    return true;
}


bool Yolov5TrtInfer::doPreprocess(const cv::Mat& cInMat)
{
    smoke_m_iInWidth = cInMat.cols;
    smoke_m_iInHeight = cInMat.rows;    
    //等比例缩放
    cv::Mat cTmpResized;       
    if( smoke_m_iInWidth >= smoke_m_iInHeight )
    {
        smoke_m_fRecoverScale = static_cast<float>(smoke_m_cModelInputSize.width) / smoke_m_iInWidth;
        cv::resize( cInMat, cTmpResized, cv::Size( smoke_m_cModelInputSize.width, smoke_m_iInHeight*smoke_m_fRecoverScale));
        smoke_m_iPadDeltaY = (smoke_m_cModelInputSize.height-smoke_m_iInHeight*smoke_m_fRecoverScale) / 2;
        smoke_m_iPadDeltaX = 0;
    }
    else
    {
        smoke_m_fRecoverScale = static_cast<float>(smoke_m_cModelInputSize.height) / smoke_m_iInHeight;
        cv::resize( cInMat, cTmpResized, cv::Size(smoke_m_iInWidth*smoke_m_fRecoverScale, smoke_m_cModelInputSize.height) );
        smoke_m_iPadDeltaX = ( smoke_m_cModelInputSize.width - smoke_m_iInWidth*smoke_m_fRecoverScale) / 2;
        smoke_m_iPadDeltaY = 0;
    }        
    //填充
    smoke_m_cPasteBoard = cv::Mat( cv::Size(smoke_m_cModelInputSize.width, smoke_m_cModelInputSize.height), CV_8UC3, cv::Scalar(114, 114, 114));
    cTmpResized.copyTo(smoke_m_cPasteBoard.rowRange(smoke_m_iPadDeltaY, smoke_m_iPadDeltaY + cTmpResized.rows).colRange(smoke_m_iPadDeltaX, smoke_m_iPadDeltaX + cTmpResized.cols));    
    cv::cvtColor(smoke_m_cPasteBoard, smoke_m_cRGBMat, cv::COLOR_BGR2RGB);
    smoke_m_cRGBMat.convertTo(smoke_m_Normalized, CV_32FC3, 1/255.);
    cv::split(smoke_m_Normalized, smoke_m_InputWrappers);
    return true;
}


bool Yolov5TrtInfer::doInference()
{
    // 输入从host拷贝到device
    auto ret = cudaMemcpyAsync(smoke_m_ArrayDevMemory[smoke_m_iInIndex], smoke_m_ArrayHostMemory[smoke_m_iInIndex], smoke_m_ArraySize[smoke_m_iInIndex], cudaMemcpyHostToDevice, smoke_m_CudaStream);
    
    // 异步推理
    auto ret1 = smoke_m_CudaContext->enqueueV2(smoke_m_ArrayDevMemory, smoke_m_CudaStream, nullptr);    
    // 输出从device拷贝到host
    ret = cudaMemcpyAsync(smoke_m_ArrayHostMemory[smoke_m_iOutIndex], smoke_m_ArrayDevMemory[smoke_m_iOutIndex], smoke_m_ArraySize[smoke_m_iOutIndex], cudaMemcpyDeviceToHost, smoke_m_CudaStream);   
    ret = cudaStreamSynchronize(smoke_m_CudaStream); 
    return true;
}


bool Yolov5TrtInfer::recoverPosInfo(std::vector<DetItem>& smokeDets, int xmin, int ymin)
{
    for( auto iter = smokeDets.begin(); iter != smokeDets.end(); ++iter)
    {   
        float x = std::max( (float)(( iter->m_cRect.x - smoke_m_iPadDeltaX ) / smoke_m_fRecoverScale + xmin), 0.f);
        float y = std::max( (float)(( iter->m_cRect.y - smoke_m_iPadDeltaY ) / smoke_m_fRecoverScale + ymin), 0.f); 
        float w = std::max( (float)(( iter->m_cRect.width ) / smoke_m_fRecoverScale), 0.f); 
        float h = std::max( (float)((iter->m_cRect.height ) / smoke_m_fRecoverScale), 0.f);
        
        iter->m_cRect.x = x;
        iter->m_cRect.y = y;
        iter->m_cRect.width = w;
        iter->m_cRect.height = h;
        
        int count = 0;
        for(auto& num:iter->Dpoints)
        {
            if(count%2==0){
                num = std::max(std::min((float)(( num - smoke_m_iPadDeltaX ) / smoke_m_fRecoverScale),(float)(smoke_m_iInWidth - 1)), 0.f) + xmin;
            }
            else{
                num = std::max(std::min((float)(( num - smoke_m_iPadDeltaY ) / smoke_m_fRecoverScale ),(float)(smoke_m_iInHeight - 1)), 0.f) + ymin;
            }
            count++;
        }
    }
    return true;  
}


bool Yolov5TrtInfer::filterByThresh(const float Thresh, std::vector<DetItem>& smokeDets)
{
    for( auto iter = smokeDets.begin(); iter != smokeDets.end(); )
    {
        if( iter->m_fScore > Thresh)
        {            
            ++iter;
        }
        else
        {
            iter = smokeDets.erase(iter);
        }
    }
    return true;
}
