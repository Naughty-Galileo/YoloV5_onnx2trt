# YOLOV5_onnx2trt
- 记录下tensorrt部署onnx的方式
- 利用两个yolov5-face的模型进行抽烟人员检测
- c++实现
- onnx-->tensorrt部署
- 利用关键点提高检测准确性
> 目前只是初版，测试数据有限，路径直接固定了，可以自己改成命令行参数输入，此外代码还有不少缺点，尤其是运行速度慢，慢慢学习优化。


##  model 1
- 利用yolov5-face检测人脸以及对应关键点
> 这个模型可以检测front_head、side_head、back_head、hand、smoke_part_box、smoke，方便后期扩展，目前这个模型只用来识别人脸，因为烟头召回率极低，故利用第二个模型检测烟头，smoke为抽烟的人，但是为了提升准确度，只用第二个模型检测到的烟头作为抽烟人员的判断依据

##  model 2
- 修改yolov5-face源代码，改为检测两个关键点的，用于检测烟头以及两个端点
> 这个模型检测smoke_part_box以及两个端点，利用这些端点进行判断筛选，提高准确性，避免误检


##  Quick Start
- 训练你的模型，并将其转换为onnx格式，为了简便，代码里onnx输出形状被我固定了，需要根据你的模型输出更改，后续有时间再改代码
- 将onnx文件放到工程目录下的model文件夹下
- 配置相关环境，opencv cuda…

```shell
mkdir build 
cd build
cmake ..
make 
```

> 参数设置比较简单
- 0测试图片 0不开启可视化 
```shell
./smoke_detect 0 0
```

- 1测试视频 0不开启可视化
```shell
./smoke_detect 0 0
```
