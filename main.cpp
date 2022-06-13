#include <cmath>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include "stdlib.h"
#include <string>
#include<opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "SampleDetector.cpp"

using namespace cv;
using namespace std;

float get_angle(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4)
{
    float vector1x = x1 - x2;
	float vector1y = y1 - y2;
	float vector2x = x3 - x4;
	float vector2y = y3 - y4;
	float t = ((vector1x)*(vector2x) + (vector1y)*(vector2y))/ (sqrt(powf(vector1x, 2.0) + powf(vector1y, 2.0))*sqrt(powf(vector2x, 2.0) + powf(vector2y, 2.0)));
	float angle = acos(t)*(180 / M_PI);
    return angle;
}

void detect_one_image(SampleDetector* detector, cv::Mat &cv_image)
{
    std::vector<SampleDetector::DetItem> detected_objects;
    auto start = chrono::high_resolution_clock::now();
    detector->ProcessImage(cv_image,detected_objects);
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout<<"Infer Time "<<diff.count()<<" s" << endl;

    std::vector<SampleDetector::DetItem> smoke_part;
    std::vector<SampleDetector::FaceObject> face;
    // std::vector<SampleDetector::DetItem> smoke;
    std::vector<SampleDetector::FaceObject> match_faces;

    for(auto &object : detected_objects){

        int xmin = object.m_cRect.x;
        int ymin = object.m_cRect.y;
        int width = object.m_cRect.width;
        int height = object.m_cRect.height;

        if(object.m_eType==0 || object.m_eType==1)
        {
            float eye_mid_x = (object.Dpoints[0] + object.Dpoints[2])/2;
            float eye_mid_y = (object.Dpoints[1] + object.Dpoints[3])/2;
            float mouth_mid_x = (object.Dpoints[6] + object.Dpoints[8])/2;
            float mouth_mid_y = (object.Dpoints[7] + object.Dpoints[9])/2;
            
            float dist = sqrt(powf((mouth_mid_x-eye_mid_x), 2.0) + powf((mouth_mid_y-eye_mid_y), 2.0));
            
            //将脸部diatance（眉心到嘴心的距离）信息加入
            SampleDetector::FaceObject temp = {object.m_fScore, object.m_eType, object.m_cRect, object.Dpoints, dist};
            face.emplace_back(temp);
        }

        // 判断这个烟头是否是误判的标志位
        bool smoke_part_flag = true;

        // 过滤smoke_part的相关参数
        float angle_front = 5;   // 正面时候，当烟头与嘴上两个关键点连线之间的角度小于该角度，则认为是误报过滤掉，能排除将牙齿误检成烟头的情况
        float angle_side = 30;   // 侧面时候，当烟头与眉心到嘴中心连线之间的角度小于该范围，则认为是误报过滤掉，能排不合理的检测结果
        float length = 1.5;      // 表示烟头长度与眉心到嘴中心的距离比值，大于比值的烟头会认为是误报而过滤掉
        float part2mouth_dist = 0.5;  // 以眉心到嘴中心的距离为基准，并计算烟头端点到嘴中心距离的比值，如果比值在该范围内则认为在嘴上
        SampleDetector::FaceObject match_face;  // 每个烟头匹配的人脸

        if(object.m_eType==3)
        {
            length = sqrt(powf((object.Dpoints[0]-object.Dpoints[2]), 2.0) + powf((object.Dpoints[1]-object.Dpoints[3]), 2.0));   
            float part_midx = object.m_cRect.x + object.m_cRect.width/2;
            float part_midy = object.m_cRect.x + object.m_cRect.width/2;

            // 匹配人脸
            float mis_distance = FLT_MAX;
            int match_index;
            for(int index=0;index<face.size();++index)
            {
                float face_midx = face[index].rect.x + face[index].rect.width/2;
                float face_midy = face[index].rect.y + face[index].rect.height/2;
                if(sqrt(powf((face_midx-part_midx), 2.0) + powf((face_midy-part_midy), 2.0)) < mis_distance)
                {
                    mis_distance = sqrt(powf((face_midx-part_midx), 2.0) + powf((face_midy-part_midy), 2.0));
                    match_index = index;
                }
            }
            match_face = face[match_index];
            length = length / match_face.dist;
                
            // radius 以眉心到嘴中心的距离为基准，并计算烟头端点到嘴中心距离的比值
            float eye_mid_x = (match_face.points[0] + match_face.points[2])/2;
            float eye_mid_y = (match_face.points[1] + match_face.points[3])/2;

            float mouth_mid_x = (match_face.points[6] + match_face.points[8])/2;
            float mouth_mid_y = (match_face.points[7] + match_face.points[9])/2;

            float part_mid_x = (object.Dpoints[0] + object.Dpoints[2])/2;
            float part_mid_y = (object.Dpoints[1] + object.Dpoints[3])/2;

            float dist_part2mouth = std::min( sqrt(powf((mouth_mid_x - object.Dpoints[0]), 2.0) + powf((mouth_mid_y - object.Dpoints[1]), 2.0)), 
                                             sqrt(powf((mouth_mid_x - object.Dpoints[2]), 2.0) + powf((mouth_mid_y - object.Dpoints[3]), 2.0)) );

            part2mouth_dist = dist_part2mouth / match_face.dist;
            
            //filterAngleRange1
            angle_front = get_angle(match_face.points[6], match_face.points[7], match_face.points[8], match_face.points[9], object.Dpoints[0], object.Dpoints[1], object.Dpoints[2], object.Dpoints[3]);
            angle_side = get_angle(eye_mid_x, eye_mid_y, mouth_mid_x, mouth_mid_y, object.Dpoints[0], object.Dpoints[1], object.Dpoints[2], object.Dpoints[3]);

            // length 烟头长度与眉心到嘴中心的距离比值
            if(length > 1.5)
            {
                smoke_part_flag = false;
            }
            // radius 以眉心到嘴中心的距离为基准
            if(part2mouth_dist > 0.5)
            {
                smoke_part_flag = false;
            }
            //filterAngleRange1
            if(match_face.m_eType == 0 && angle_front < 5)
            {
                smoke_part_flag = false;
            }
            
            //filterAngleRange2
            if(match_face.m_eType == 1 && angle_side < 30)
            {
                smoke_part_flag = false;
            }
            
            
            // 正脸的时候，烟头这条线必须分割人脸
            if(match_face.m_eType == 0)
            {
                float x_min = std::min(match_face.points[0], std::min(match_face.points[2], std::min(match_face.points[4], std::min(match_face.points[6], match_face.points[8]))));
                float y_min = std::min(match_face.points[1], std::min(match_face.points[3], std::min(match_face.points[5], std::min(match_face.points[7], match_face.points[9]))));
                float x_max = std::max(match_face.points[0], std::max(match_face.points[2], std::max(match_face.points[4], std::max(match_face.points[6], match_face.points[8]))));
                float y_max = std::max(match_face.points[1], std::max(match_face.points[3], std::max(match_face.points[5], std::max(match_face.points[7], match_face.points[9]))));
                if( (object.Dpoints[2] - object.Dpoints[0])==0 )
                {
                    if( (match_face.points[0] - object.Dpoints[0])*(match_face.points[2] - object.Dpoints[0])>0 && (match_face.points[6] - object.Dpoints[0])*(match_face.points[8] - object.Dpoints[0])>0 )
                    {
                        smoke_part_flag = false;
                    }
                }
                else if( (object.Dpoints[3] - object.Dpoints[1])==0 )
                {
                    if(( (y_min - object.Dpoints[1])*(y_max - object.Dpoints[1]) > 0 ))
                    {
                        smoke_part_flag = false;
                    }
                }
                else if(
                    ((x_min - object.Dpoints[0])/(object.Dpoints[2] - object.Dpoints[0]) - (y_min - object.Dpoints[1])/(object.Dpoints[3] - object.Dpoints[1]))*
                    ((x_max - object.Dpoints[0])/(object.Dpoints[2] - object.Dpoints[0]) - (y_max - object.Dpoints[1])/(object.Dpoints[3] - object.Dpoints[1])) > 0
                    &&
                    ((x_min - object.Dpoints[0])/(object.Dpoints[2] - object.Dpoints[0]) - (y_max - object.Dpoints[1])/(object.Dpoints[3] - object.Dpoints[1]))*
                    ((x_max - object.Dpoints[0])/(object.Dpoints[2] - object.Dpoints[0]) - (y_min - object.Dpoints[1])/(object.Dpoints[3] - object.Dpoints[1])) > 0
                )
                {
                    smoke_part_flag = false;
                }
            }
            if(smoke_part_flag)
            {
                smoke_part.emplace_back(object);
                match_faces.emplace_back(match_face);
            }
        }
        if(object.m_eType == 5)
        {
            // smoke.emplace_back(object);
            continue;
        }

        if(object.m_eType == 0 || object.m_eType == 1)
        {
                int colors[5][3] = {{255,0,0},{0,255,0},{0,0,255},{255,255,0},{0,255,255}};
                for(int i=0;i<5;++i)
                {
                    cv::Point point(object.Dpoints[2*i], object.Dpoints[2*i+1]);
                    int radiusCircle = 1;
                    cv::Scalar colorCircle1(colors[i][0], colors[i][1], colors[i][2]); // (B, G, R)
                    int thicknessCircle1 = 1;
                    cv::circle(cv_image, point, radiusCircle, colorCircle1, thicknessCircle1);
                }
        }
    }

    int Width = cv_image.cols;
    int Height = cv_image.rows;
    if(!smoke_part.empty())
    {
        for(int i=0;i<smoke_part.size();i++)
        {
            SampleDetector::DetItem part = smoke_part[i];
            SampleDetector::FaceObject match_face = match_faces[i];      
            int smoke_x = max(0,match_face.rect.x - match_face.rect.width);
            int smoke_y = match_face.rect.y;
            int smoke_width = min(3*match_face.rect.width, Width-smoke_x-1);
            int smoke_height = min(2*match_face.rect.height, Height-smoke_y-1);
            
            rectangle(cv_image, Point(smoke_x, smoke_y), Point(smoke_x+smoke_width, smoke_y+smoke_height), cv::Scalar(0, 0, 255), 1);
            std::stringstream ss;
            ss << "smoke";
            cv::Point textLeftBottom(smoke_x, smoke_y);
            cv::putText(cv_image, ss.str(), textLeftBottom, FONT_HERSHEY_COMPLEX, 0.6,
                        cv::Scalar(0, 0, 255), 1, LINE_8);
        }
        std::stringstream ss;
        ss << "Warning Warning";
        cv::Point textLeftBottom(10, 50);
        cv::putText(cv_image, ss.str(), textLeftBottom, FONT_HERSHEY_COMPLEX, 1,
                        cv::Scalar(0, 0, 255), 1, LINE_8);
    }
}

int main(int argc, char const *argv[])
{
    SampleDetector* detector = new SampleDetector;
    string model_path = "../model/face5-best.onnx";
    detector->Init(model_path);

    if(atoi(argv[1]) == 0)
    {
        cout<< "image"<<endl;
        cv::Mat src;
        src = cv::imread("../data/test.jpg");
        cv::Mat osrc = src.clone();
        detect_one_image(detector, osrc);

        if(argc>=3 && atoi(argv[2]) == 1)
        {
            cv::imshow("result",osrc);
            cv::waitKey(10);
        }
        cv::imwrite("../result.jpg", osrc);
    }

    else if(atoi(argv[1]) == 1)
    {
        cout<< "video"<<endl;
        cv::VideoCapture capture;
        cv::Mat frame;
        string video_path = "../data/test.mp4";
        capture.open(video_path);
        if (!capture.isOpened()) {
            printf("could not read this video file...\n");
            return -1;
	    }
        Size size = Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
        string out_path = "../result.mp4";
        VideoWriter writer(out_path, 0x7634706d, capture.get(CAP_PROP_FPS), size, true);
        
        while (capture.read(frame))
        {
            cv::Mat src = frame.clone();
            detect_one_image(detector, src);
            writer.write(src);
            if(argc>=3 && atoi(argv[2]) == 1)
            {
                imshow("output", src);
                waitKey(10);
            }
        }
        capture.release();
        writer.release();
        waitKey(0);
    }

    else
    {
        cout<< "video"<<endl;
        cv::VideoCapture capture;
        cv::Mat frame;
        capture.open(0);
        if (!capture.isOpened()) {
            printf("could not read this video file...\n");
            return -1;
	    }
        Size size = Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
        string out_path = "../result.mp4";
        VideoWriter writer(out_path, 0x7634706d, capture.get(CAP_PROP_FPS), size, true);
        
        while (capture.read(frame))
        {
            cv::Mat src = frame.clone();
            detect_one_image(detector, src);
            writer.write(src);
            imshow("output", src);
            waitKey(10);
        }
        capture.release();
        writer.release();
        waitKey(0);
    }
    
    return 0;
}

