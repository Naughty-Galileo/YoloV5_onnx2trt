#include "SampleDetector.hpp"

#define IN_NAME "images"
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

        // FP16 
        config->setFlag(BuilderFlag::kFP16);

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
    
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInIndex] + 2 * sizeof(float) * dims_i.d[2] * dims_i.d[3]);
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInIndex] + sizeof(float) * dims_i.d[2] * dims_i.d[3] );
    m_InputWrappers.emplace_back(dims_i.d[2], dims_i.d[3], CV_32FC1, m_ArrayHostMemory[m_iInIndex]);
    
    
    m_ArraySize[m_iInIndex] = size *sizeof(float);
    
    //output
    dims_i = m_CudaEngine->getBindingDimensions(m_iOutIndex);
    cout << "output dims "<< dims_i.d[0] << " " << dims_i.d[1] << " " << dims_i.d[2] << endl;
    m_iBoxNums = dims_i.d[1];
    mClasses = dims_i.d[2]-5;
    size = dims_i.d[0] * dims_i.d[1] * dims_i.d[2];
    cudaMalloc(&m_ArrayDevMemory[m_iOutIndex], size * sizeof(float));
    m_ArrayHostMemory[m_iOutIndex] = malloc( size * sizeof(float));
    m_ArraySize[m_iOutIndex] = size *sizeof(float);
    
    cudaStreamCreate(&m_CudaStream);
    m_cPasteBoard = cv::Mat(m_cModelInputSize, CV_8UC3, cv::Scalar(128, 128, 128));
    cout << "Init Done."<<endl;
    m_bUninit = false;
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
    cout<<"UnInit"<<endl;
}

bool SampleDetector::ProcessImage(const cv::Mat &cv_image, std::vector<DetItem> &vecDets) {
    
    img_w = cv_image.cols;
    img_h = cv_image.rows;
    
    doPreprocess(cv_image);
    doInference();
    
    float* pData = (float*)m_ArrayHostMemory[m_iOutIndex];
    
    int iIndex = 0;
    float fConf = 0.f, x = 0.f, y = 0.f, w = 0.f, h = 0.f;
    int iBestClassId = 0;
    
    for(int i=0;i<m_iBoxNums;++i)
    {
        iIndex = i * (mClasses+5) + 4;
        fConf = pData[iIndex];
        if (fConf > 0.05)    // 0.05
        {
            h = pData[--iIndex];
            w = pData[--iIndex];
            y = pData[--iIndex];
            x = pData[--iIndex];
            
            iIndex += 5;
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

            vecDets.push_back( {x- w/2.0, y-h/2.0, x + w/2.0, y + h/2.0, fConf, iBestClassId - iIndex});                 
        }
    }
    // 恢复位置到原图
    recoverPosInfo(vecDets);
    // nms
    doNms(vecDets);
    
    // 阈值过滤
    filter(vecDets);
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
    m_cRGBMat.convertTo(m_Normalized, CV_32FC3, 1.0/255.0);
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
        float x1 = std::max(std::min((float)((iter->x1 - m_iPadDeltaX)/m_fRecoverScale), (float)(img_w - 1)), 0.f);
        float y1 = std::max(std::min((float)((iter->y1 -m_iPadDeltaY)/ m_fRecoverScale),(float)(img_h - 1)), 0.f);
        float x2 = std::max(std::min((float)((iter->x2 - m_iPadDeltaX)/m_fRecoverScale), (float)(img_w - 1)), 0.f);
        float y2 = std::max(std::min((float)((iter->y2 - m_iPadDeltaY)/ m_fRecoverScale),(float)(img_h - 1)), 0.f);

        iter->x1 = x1;
        iter->y1 = y1;
        iter->x2 = x2;
        iter->y2 = y2;
    }
    return true;  
}


bool SampleDetector::filter(std::vector<DetItem>& vecDets)
{
    for( auto iter = vecDets.begin(); iter != vecDets.end(); )
    {
        float fThresh = mConf_thresh;
        if( iter->score > fThresh)
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
    std::sort(vecDets.begin(),vecDets.end(),[](const DetItem &det1, const DetItem &det2){return det1.score > det2.score;});
    
    for (int i = 0; i < vecDets.size(); ++i)
    {
        if (vecDets[i].score < 0.1f)
        {
            continue;
        }
        
        for (int j = i+1; j < vecDets.size(); ++j)
        {
            if(vecDets[i].label == vecDets[j].label){

                cv::Rect rect1 = cv::Rect{vecDets[i].x1, vecDets[i].y1, vecDets[i].x2 - vecDets[i].x1, vecDets[i].y2 - vecDets[i].y1};
                cv::Rect rect2 = cv::Rect{vecDets[j].x1, vecDets[j].y1, vecDets[j].x2 - vecDets[i].x1, vecDets[j].y2 - vecDets[j].y1};

                if ( boxIOU(rect1, rect2) >= mIou_thresh ) 
                {
                    vecDets[j].score = 0;
                }
            }     
        }
    }
    
    for( auto iter = vecDets.begin(); iter != vecDets.end(); )
    {
        if( iter->score < 0.1f )
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
