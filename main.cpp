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

void detect_one_image(SampleDetector* detector, cv::Mat &cv_image)
{
    vector<std::string> classnames = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", 
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", 
    "baseball glove", "skateboard","surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

    std::vector<SampleDetector::DetItem> detected_objects;
    auto start = chrono::high_resolution_clock::now();

    detector->ProcessImage(cv_image, detected_objects);

    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout<<"Infer Time "<<diff.count()<<" s" << std::endl;

    for(auto &object : detected_objects)
    {

        int xmin = object.x1;
        int ymin = object.y1;
        int xmax = object.x2;
        int ymax = object.y2;

        rectangle(cv_image, Point(xmin, ymin), Point(xmax, ymax), cv::Scalar(0, 0, 255), 1);
        std::stringstream ss;
        ss << classnames[object.label];
        cv::Point textLeftBottom(xmin, ymin);
        cv::putText(cv_image, ss.str(), textLeftBottom, FONT_HERSHEY_COMPLEX, 0.6,
                    cv::Scalar(0, 0, 255), 1, LINE_8);
    }
}

int main(int argc, char const *argv[])
{
    SampleDetector* detector = new SampleDetector;
    std::string model_path = "./model/yolov5s.onnx";
    detector->Init(model_path);

    if(atoi(argv[1]) == 0)
    {
        cout<< "image... ..."<<endl;
        cv::Mat src;
        src = cv::imread("./images/bus.jpg");
        cv::Mat osrc = src.clone();
        detect_one_image(detector, osrc);

        cv::imwrite("./assert/result.jpg", osrc);
    }

    // else if(atoi(argv[1]) == 1)
    // {
    //     std::cout<< "video... ..."<<std::endl;

    //     cv::VideoCapture capture;
    //     cv::Mat frame;
    //     string video_path = "../data/test.mp4";
    //     capture.open(video_path);
    //     if (!capture.isOpened()) {
    //         printf("could not read this video file...\n");
    //         return -1;
	//     }
    //     Size size = Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
    //     string out_path = "../result.mp4";
    //     VideoWriter writer(out_path, 0x7634706d, capture.get(CAP_PROP_FPS), size, true);
        
    //     while (capture.read(frame))
    //     {
    //         cv::Mat src = frame.clone();
    //         detect_one_image(detector, src);
    //         writer.write(src);
    //         if(argc>=3 && atoi(argv[2]) == 1)
    //         {
    //             imshow("output", src);
    //             waitKey(10);
    //         }
    //     }
    //     capture.release();
    //     writer.release();
    //     waitKey(0);
    // }

    // else
    // {
    //     cout<< "camera... ..."<<endl;
    //     cv::VideoCapture capture;
    //     cv::Mat frame;
    //     capture.open(0);
    //     if (!capture.isOpened()) {
    //         printf("could not read this video file...\n");
    //         return -1;
	//     }
    //     Size size = Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
    //     string out_path = "../result.mp4";
    //     VideoWriter writer(out_path, 0x7634706d, capture.get(CAP_PROP_FPS), size, true);
        
    //     while (capture.read(frame))
    //     {
    //         cv::Mat src = frame.clone();
    //         detect_one_image(detector, src);
    //         writer.write(src);
    //         imshow("output", src);
    //         waitKey(10);
    //     }
    //     capture.release();
    //     writer.release();
    //     waitKey(0);
    // }
    
    return 0;
}

