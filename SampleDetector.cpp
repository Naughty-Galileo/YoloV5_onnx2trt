#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <glog/logging.h>

#include "SampleDetector.hpp"
#include "Yolov5TrtInfer.hpp"

#include "NvOnnxParser.h"
#include "./logging.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "NvInfer.h"

#define IN_NAME "input"
#define OUT_NAME "output"

using namespace nvinfer1;
using namespace nvonnxparser;

float boxIOU(const cv::Rect2d &rect1, const cv::Rect2d &rect2)
{
    cv::Rect2d intersecRect =  rect1 & rect2;
    return intersecRect.area() / ( rect1.area() + rect2.area() - intersecRect.area() );
}


bool exists(const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

SampleDetector::SampleDetector() 
{
    
}
SampleDetector::~SampleDetector()
{
    UnInit();   
}


bool SampleDetector::Init(const std::string& strModelName) {
    cout << "Loading model..." <<endl;
    
    std::string strTrtName = strModelName;
    size_t sep_pos = strTrtName.find_last_of(".");
    strTrtName = strTrtName.substr(0, sep_pos) + ".trt";
    Logger gLogger;
    if(!exists(strTrtName))
    {
        // Logger gLogger;
        IRuntime* m_CudaRuntime = createInferRuntime(gLogger);
        IBuilder* builder = createInferBuilder(gLogger);
        builder->setMaxBatchSize(1);

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
        parser->parseFromFile(strModelName.c_str(), static_cast<int>(ILogger::Severity::kWARNING));

        IBuilderConfig* config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1ULL << 30);

        m_CudaEngine = builder->buildEngineWithConfig(*network, *config);
        m_CudaContext = m_CudaEngine->createExecutionContext();
        
        IHostMemory *gieModelStream = m_CudaEngine->serialize();
        std::string serialize_str;
        std::ofstream serialize_output_stream;
        serialize_str.resize(gieModelStream->size());   
        memcpy((void*)serialize_str.data(), gieModelStream->data(), gieModelStream->size());
        serialize_output_stream.open(strTrtName);
        serialize_output_stream<<serialize_str;
        serialize_output_stream.close();
        cout << "serialize model success " <<endl;
        
        parser->destroy();
        network->destroy();
        config->destroy();
        builder->destroy();
    }
    else{
        // Logger gLogger;
        IRuntime* runtime = createInferRuntime(gLogger);
        
        std::string cached_path = strTrtName;
        std::ifstream fin(cached_path);
        std::string cached_engine = "";
        while (fin.peek() != EOF){ 
                std::stringstream buffer;
                buffer << fin.rdbuf();
                cached_engine.append(buffer.str());
        }
        fin.close();
        m_CudaEngine = runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
        m_CudaContext = m_CudaEngine->createExecutionContext();
        runtime->destroy();
        cout << "deserialize model success "<<endl;
    }
    
    // 分配输入输出的空间,DEVICE侧和HOST侧
    m_iInIndex = m_CudaEngine->getBindingIndex(IN_NAME);
    m_iOutIndex = m_CudaEngine->getBindingIndex(OUT_NAME);
    
    Dims dims_i = m_CudaEngine->getBindingDimensions(m_iInIndex);
    cout << "input dims " << dims_i.d[0] << " " << dims_i.d[1] << " " << dims_i.d[2] << " " << dims_i.d[3] << endl;
    
    int size = dims_i.d[0] * dims_i.d[1] * dims_i.d[2] * dims_i.d[3];
    m_cModelInputSize = cv::Size(dims_i.d[3], dims_i.d[2]);
    
    cudaMalloc(&m_ArrayDevMemory[m_iInIndex], size * sizeof(float));
    m_ArrayHostMemory[m_iInIndex] = malloc(size * sizeof(float));
    
    //方便NHWC到NCHW的预处理
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInIndex]);
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInIndex] + sizeof(float) * dims_i.d[2] * dims_i.d[3] );
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInIndex] + 2 * sizeof(float) * dims_i.d[2] * dims_i.d[3]);
    m_ArraySize[m_iInIndex] = size *sizeof(float);
    
    //output
    dims_i = m_CudaEngine->getBindingDimensions(m_iOutIndex);
    cout << "output dims "<< dims_i.d[0] << " " << dims_i.d[1] << " " << dims_i.d[2] << endl;
    mClasses = dims_i.d[2]-15;
    size = dims_i.d[0] * dims_i.d[1] * dims_i.d[2];
    cudaMalloc(&m_ArrayDevMemory[m_iOutIndex], size * sizeof(float));
    m_ArrayHostMemory[m_iOutIndex] = malloc( size * sizeof(float));
    m_ArraySize[m_iOutIndex] = size *sizeof(float);
    
    cudaStreamCreate(&m_CudaStream);
    m_cPasteBoard = cv::Mat(m_cModelInputSize, CV_8UC3, cv::Scalar(128, 128, 128));

    smoke_detector = new Yolov5TrtInfer();
    cout << "Init Done."<<endl;
}

bool SampleDetector::UnInit() {
    if(m_bUninit == true)
    {
        return false;
    }
    for(auto &p:m_ArrayDevMemory)
    {
        cudaFree(p);
        p = nullptr;
    }
    for(auto &p: m_ArrayHostMemory)
    {        
        free(p);
        p = nullptr;        
    } 
    
    m_CudaContext->destroy();
    m_CudaEngine->destroy();
    cudaStreamDestroy(m_CudaStream);
    m_bUninit = true;
    delete smoke_detector;
    cout<<"UnInit"<<endl;
}

bool SampleDetector::ProcessImage(const cv::Mat &cv_image, std::vector<DetItem> &vecDets) {
    
    img_w = cv_image.cols;
    img_h = cv_image.rows;
    
    doPreprocess(cv_image);
    doInference();
    
    float* pData = (float*)m_ArrayHostMemory[m_iOutIndex];
    int iNumClass = mClasses;
    
    int iIndex = 0;
    float fConf = 0, x = 0, y = 0, w = 0, h = 0, r_x = 0, r_y = 0;
    int iBestClassId = 0;
    for(int i=0;i<25200;++i)
    {
        iIndex = i * 21 + 4;
        fConf = pData[iIndex];
        if (fConf > 0.05)    // 0.05
        {
            h = pData[--iIndex];
            w = pData[--iIndex];
            y = pData[--iIndex];
            x = pData[--iIndex];
            
            iIndex += 5;
            std::vector<float> points;
            for(int j=0;j<5;j++)
            {
                points.emplace_back((float)pData[iIndex++]);
                points.emplace_back((float)pData[iIndex++]);
            }

            iBestClassId = iIndex;
            for (int c = 1; c < mClasses; c++)
            {
                if ( pData[iIndex + c] > pData[iBestClassId])
                {
                    iBestClassId = iIndex + c;
                }
            }
            fConf = fConf*pData[iBestClassId];
            if (fConf < 0.05) continue;         

            vecDets.push_back( {cv::Rect2d(x-w/2, y-h/2, w, h), fConf, iBestClassId - iIndex, points});                 
        }
    }
    
    filter(vecDets);
    // 恢复位置到原图
    recoverPosInfo(vecDets);
    //nms
    doNms(vecDets);
    int count = vecDets.size();
    
    int Width = cv_image.cols;
    int Height = cv_image.rows;
    
    for(int index=0;index<count;index++)
    {
        if(vecDets[index].m_eType==0 || vecDets[index].m_eType==1)
        {
            int new_xmin = std::max((int)(vecDets[index].m_cRect.x - vecDets[index].m_cRect.width), 1);
            int new_ymin = std::max((int)(vecDets[index].m_cRect.y), 1);
            int new_xmax = std::min((int)(vecDets[index].m_cRect.x + 2*vecDets[index].m_cRect.width), Width-1);
            int new_ymax = std::min((int)(vecDets[index].m_cRect.y + 2*vecDets[index].m_cRect.height), Height-1);

            if(new_xmax-new_xmin>mFace_size && new_ymax-new_ymin>mFace_size)
            {
                cv::Rect rect(new_xmin, new_ymin, new_xmax-new_xmin, new_ymax-new_ymin);
                
                cv::Mat crop_image = cv_image(rect);

                std::vector<Yolov5TrtInfer::DetItem> smokeDets;

                smoke_detector->runDetect(crop_image, mSmoke_part_box_thresh, mSmokePartArea, smokeDets, new_xmin, new_ymin);

                for(auto smoke_part:smokeDets)
                {
                    cout << "Found smoke_part_box" << endl;
                    vecDets.push_back( {smoke_part.m_cRect, smoke_part.m_fScore, 3, smoke_part.Dpoints} );
                }
            }
        } 
    }
    //阈值过滤
    filterByThresh(vecDets);
    std::sort(vecDets.begin(), vecDets.end(), [](const DetItem &det1, const DetItem &det2){return det1.m_eType < det2.m_eType;});
}

bool SampleDetector::doPreprocess(const cv::Mat& cInMat)
{
    m_iInWidth = cInMat.cols;
    m_iInHeight = cInMat.rows;    
    //等比例缩放
    cv::Mat cTmpResized;       
    if( m_iInWidth >= m_iInHeight)
    {
        m_fRecoverScale = static_cast<float>(m_cModelInputSize.width) / m_iInWidth;
        cv::resize( cInMat, cTmpResized, cv::Size( m_cModelInputSize.width, m_iInHeight * m_fRecoverScale));
        m_iPadDeltaY = (m_cModelInputSize.height - m_iInHeight * m_fRecoverScale) / 2;
        m_iPadDeltaX = 0;
    }
    else
    {
        m_fRecoverScale = static_cast<float>(m_cModelInputSize.height) / m_iInHeight;
        cv::resize(cInMat,cTmpResized, cv::Size( m_iInWidth * m_fRecoverScale, m_cModelInputSize.height));
        m_iPadDeltaX = (m_cModelInputSize.width - m_iInWidth * m_fRecoverScale) / 2;
        m_iPadDeltaY = 0;
    }        
    
    //填充
    m_cPasteBoard = cv::Mat( cv::Size(m_cModelInputSize.width, m_cModelInputSize.height), CV_8UC3, cv::Scalar(114, 114, 114));
    cTmpResized.copyTo(m_cPasteBoard.rowRange(m_iPadDeltaY, m_iPadDeltaY + cTmpResized.rows).colRange(m_iPadDeltaX, m_iPadDeltaX + cTmpResized.cols));
    
    cv::cvtColor(m_cPasteBoard, m_cRGBMat, cv::COLOR_BGR2RGB);
    m_cRGBMat.convertTo(m_Normalized, CV_32FC3, 1/255.);
    cv::split(m_Normalized, m_InputWrappers);
    return true;
}

bool SampleDetector::doInference()
{
    // 输入从host拷贝到device
    auto ret = cudaMemcpyAsync(m_ArrayDevMemory[m_iInIndex], m_ArrayHostMemory[m_iInIndex], m_ArraySize[m_iInIndex], cudaMemcpyHostToDevice, m_CudaStream);
    // 异步推理
    auto ret1 = m_CudaContext->enqueueV2(m_ArrayDevMemory, m_CudaStream, nullptr);    
    // 输出从device拷贝到host
    ret = cudaMemcpyAsync(m_ArrayHostMemory[m_iOutIndex], m_ArrayDevMemory[m_iOutIndex], m_ArraySize[m_iOutIndex], cudaMemcpyDeviceToHost, m_CudaStream);   
    ret = cudaStreamSynchronize(m_CudaStream); 
    return true;
}

bool SampleDetector::recoverPosInfo(std::vector<DetItem>& vecDets)
{
    for( auto iter = vecDets.begin(); iter != vecDets.end(); ++iter)
    {       
        float x = std::max(std::min((float)((iter->m_cRect.x-m_iPadDeltaX)/m_fRecoverScale),(float)(img_w - 1)), 0.f);
        float y = std::max(std::min((float)((iter->m_cRect.y-m_iPadDeltaY)/ m_fRecoverScale),(float)(img_h - 1)), 0.f);
        float w = std::max(std::min((float)((iter->m_cRect.width)/m_fRecoverScale),(float)(img_w - x - 1)), 0.f);
        float h = std::max(std::min((float)((iter->m_cRect.height)/m_fRecoverScale),(float)(img_h - y - 1)), 0.f);
        
        iter->m_cRect.x = x;
        iter->m_cRect.y = y;
        iter->m_cRect.width = w;
        iter->m_cRect.height = h;
        
        int count = 0;
        if(iter->m_eType==0 || iter->m_eType==1)
        {
            for(auto& num:iter->Dpoints)
            {
                if(count%2==0){
                    num = std::max(std::min((float)((num-m_iPadDeltaX)/m_fRecoverScale),(float)(img_w - 1)), 0.f);
                    if(num<x-10 || num>x+w+10)
                    {
                        iter->m_fScore = 0.f;
                    }
                }
                else{
                    num = std::max(std::min((float)(( num - m_iPadDeltaY ) / m_fRecoverScale),(float)(img_h - 1)), 0.f);
                    if(num<y-10 || num>y+h+10)
                    {
                        iter->m_fScore = 0.f;
                    }
                }
                count++;
            }
        }
    }
    return true;  
}


bool SampleDetector::filter(std::vector<DetItem>& vecDets)
{
    for( auto iter = vecDets.begin(); iter != vecDets.end(); )
    {
        float fThresh = mConf_thresh;
        if(iter->m_eType == 0){
            fThresh = mFront_head_thresh;
        }
        else if(iter->m_eType == 1){
            fThresh = mSide_head_thresh;
        }
        else if(iter->m_eType == 2){
            fThresh = mBack_head_thresh;
        }
        else if(iter->m_eType == 4){
            fThresh = mHand_thresh;
        }
        else if(iter->m_eType == 3){
            fThresh = mSmoke_part_box_thresh;
            iter = vecDets.erase(iter);
            continue;
        }
        else if(iter->m_eType == 5){
            fThresh = mSmoke_thresh;
        }
        if( iter->m_fScore > fThresh)
        {            
            ++iter;
        }
        else
        {
            iter = vecDets.erase(iter);
        }
    }
    return true;
}


bool SampleDetector::filterByThresh(std::vector<DetItem>& vecDets)
{
    for( auto iter = vecDets.begin(); iter != vecDets.end(); )
    {
        float fThresh = mConf_thresh; 
        if(iter->m_eType == 0){
            fThresh = mFront_head_thresh;
        }
        else if(iter->m_eType == 1){
            fThresh = mSide_head_thresh;
        }
        else if(iter->m_eType == 2){
            fThresh = mBack_head_thresh;
        }
        else if(iter->m_eType == 3){
            fThresh = mSmoke_part_box_thresh;
        }
        else if(iter->m_eType == 4){
            fThresh = mHand_thresh;
        }
        else if(iter->m_eType == 5){
            fThresh = mSmoke_thresh;
        }
        
        if( iter->m_fScore > fThresh)
        {            
            ++iter;
        }
        else
        {
            iter = vecDets.erase(iter);
        }
    }
    return true;
}

bool SampleDetector::doNms(std::vector<DetItem>& vecDets)
{
    std::sort(vecDets.begin(),vecDets.end(),[](const DetItem &det1, const DetItem &det2){return det1.m_fScore > det2.m_fScore;});
    bool head = false;
    
    for (int i = 0; i < vecDets.size(); ++i)
    {
        if (vecDets[i].m_fScore < 0.005f)
        {
            continue;
        }
        if(vecDets[i].m_eType==0 || vecDets[i].m_eType==1)
        {
            head = true;
            if(vecDets[i].m_cRect.area()< mFace_size*mFace_size){
                vecDets[i].m_fScore = 0;
            }
        }
        
        for (int j = i+1; j < vecDets.size(); ++j)
        {
            if(head && (vecDets[j].m_eType==0 ||  vecDets[j].m_eType==1))
            {
                if(vecDets[i].m_cRect == (vecDets[j].m_cRect&vecDets[i].m_cRect))
                {
                    vecDets[i].m_fScore = 0;
                }
                else if ( boxIOU(vecDets[i].m_cRect, vecDets[j].m_cRect) >= mIou_thresh ) 
                {
                    vecDets[j].m_fScore = 0;
                }
                else if(vecDets[j].m_cRect == (vecDets[j].m_cRect&vecDets[i].m_cRect))
                {
                    vecDets[j].m_fScore = 0;
                }
            }
            else if(vecDets[i].m_eType == vecDets[j].m_eType){
                if(vecDets[i].m_cRect == (vecDets[j].m_cRect&vecDets[i].m_cRect))
                {
                    vecDets[i].m_fScore = 0;
                }
                else if ( boxIOU(vecDets[i].m_cRect, vecDets[j].m_cRect) >= mIou_thresh ) 
                {
                    vecDets[j].m_fScore = 0;
                }
                else if(vecDets[j].m_cRect == (vecDets[j].m_cRect&vecDets[i].m_cRect))
                {
                    vecDets[j].m_fScore = 0;
                }
            }     
        }
    }
    
    for( auto iter = vecDets.begin(); iter != vecDets.end(); )
    {
        if( iter->m_fScore < 0.001f )
        {
            iter = vecDets.erase(iter);
        }
        else
        {
            iter++;
        }
    }
    return true;
}