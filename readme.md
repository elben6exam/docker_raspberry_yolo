## 要求

1. 树梅派 4b (ubuntu20.04 server)

2. linux 操作系统 电脑

## 交叉编译

    cd arm_docker
    docker image build -t 
    
    arm-plugin -f dockerfiles/Dockerfile.RPiRPi64_focal .

    mkdir build
    docker container run --rm -ti -v $PWD/build:/arm_cpu_plugin arm-plugin


在build文件中产生 **OV_ARM_package.tar.gz**文件，其中蕴含ncnn,MNN,opencv,openvino库。将其解压至树梅派系统

在armcpu_package/extra/中包含MNN和ncnn的库，请根据自己需要添加到系统路径中。

## 部署模型

提供了MNN，ncnn的c++图像识别代码。请注意，读取yolo架构的模型，需要获取3个模型中的permute层的具体名字，当套用不同的yolo模型时，可能会产生**段错误**,因为不同的模型转换方式会导致输出层名字的不同，请根据源代码和模型具体进行修改。

### MNN

1. 确定模型输出层和anchor
2. 读取视频
3. 创建引擎
4. 循环读取图片
    
    4.1 图像预处理
    4.2 推理
    4.3 从指定输出层读取数据
    4.4 标记图像
    4.5 输出图像
 

