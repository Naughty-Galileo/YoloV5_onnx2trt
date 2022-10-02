#include <cmath>
#include <string>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include "stdlib.h"
#include <sys/stat.h>
#include<opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "SampleDetector.cpp"


using namespace cv;


int check_filetype(const string &filename)
{
    int filetype = -1; //0:image; 1:video

    std::size_t found = filename.rfind('.');
    if (found != std::string::npos)
    {
        string strExt = filename.substr(found);        
        std::transform(strExt.begin(), strExt.end(), strExt.begin(), ::tolower);
        if (strExt.compare(".jpg") == 0 ||
            strExt.compare(".jpeg") == 0 ||
            strExt.compare(".png") == 0 ||
            strExt.compare(".bmp") == 0 )
        {
            // std::cout << "file type is image" << std::endl;
            filetype = 0;
        }
        if (strExt.compare(".mp4") == 0 ||
            strExt.compare(".avi") == 0 ||
            strExt.compare(".flv") == 0 ||
            strExt.compare(".mkv") == 0 ||
            strExt.compare(".wmv") == 0 ||
            strExt.compare(".rmvb") == 0)
        {
            // std::cout << "file type is video" << std::endl;
            filetype = 1;
        }        
    }
    return filetype;
}

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

int main(int argc, char *argv[])
{
    int opt;
    const char *str = "i:o:";

    std::string input_file;
    std::string output_file;
    int file_type = -1;


    while((opt = getopt(argc, argv, str)) != -1)
    {
        switch (opt)
        {
            case 'i':
            {
                input_file = string(optarg);
                break;
            }
            case 'o':
            {
                output_file = string(optarg);
                break;
            }
            default: break;
        }
    }

    if(!input_file.empty())
    {
        file_type = check_filetype(input_file);
    }

    if(output_file.empty())
    {
        switch (file_type)
        {
        case 0:
            output_file = "./outputs/result.jpg";
            break;
        default:
            output_file = "./outputs/result.mp4";
            break;
        }
    }

    if(!input_file.empty() && !output_file.empty()){
        assert(check_filetype(input_file)==check_filetype(output_file));
    }

    SampleDetector* detector = new SampleDetector;
    std::string model_path = "./model/yolov5s.onnx";
    detector->Init(model_path);

    if( file_type == 0)
    {
        cout<< "file type is image"<<endl;
        cv::Mat src;
        src = cv::imread(input_file);
        cv::Mat osrc = src.clone();
        detect_one_image(detector, osrc);

        cv::imwrite(output_file, osrc);
    }

    else if( file_type == 1)
    {
        std::cout<< "file type is video"<<std::endl;

        cv::VideoCapture capture;
        cv::Mat frame;
        capture.open(input_file);
        if (!capture.isOpened()) {
            printf("could not read this video file...\n");
            return -1;
	    }
        
        Size size = Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
        VideoWriter writer(output_file, 0x7634706d, capture.get(CAP_PROP_FPS), size, true);
        
        while (capture.read(frame))
        {
            cv::Mat src = frame.clone();
            detect_one_image(detector, src);
            writer.write(src);
        }
        capture.release();
        writer.release();
        waitKey(0);
    }

    else
    {
        std::cout<< "file type is camera"<<std::endl;
        cv::VideoCapture capture;
        cv::Mat frame;
        capture.open(0);
        if (!capture.isOpened()) {
            printf("could not read camera...\n");
            return -1;
	    }
        Size size = Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
        VideoWriter writer(output_file, 0x7634706d, capture.get(CAP_PROP_FPS), size, true);
        
        while (capture.read(frame))
        {
            cv::Mat src = frame.clone();
            detect_one_image(detector, src);
            writer.write(src);
        }
        capture.release();
        writer.release();
        waitKey(0);
    }
    
    return 0;
}

