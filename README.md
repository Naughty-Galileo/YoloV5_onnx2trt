# YoloV5_onnx2trt
> onnx->tensorrt c++ 部署
> 支持图片、视频、摄像头

##  Quick Start
- 训练模型
- 导出为onnx文件
    > 在使用yolov5官方源码导出onnx模型时**只支持tensort主版本为8的推理** \
    > 若tensorrt主版本为7时需要修改导出代码，导出代码见export.py  
- 将onnx文件放到工程目录下的model文件夹下
- 配置相关环境，opencv,cuda …
- 更改classnames,mIou_thresh,mConf_thresh
- 编译
    ```shell
    mkdir build 
    cd build
    cmake ..
    make 
    ```
- 运行 -i 输入路径 -o 输出路径
     ```
  ./build/demo -i ./images/bus.jpg -o ./assert/bus_result.jpg
     ```
## Demo
![avatar](./assert/bus_result.jpg)


