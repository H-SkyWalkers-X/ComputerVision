# 手写数字识别系统

基于DenseNet的手写数字识别系统，支持单个数字识别和连续数字（学号）识别。该系统使用PyTorch框架实现，并提供了友好的Web界面。

## 项目结构

```
Experience/Task-3/
├── front.py              # Web界面实现（基于Streamlit）
├── model.py              # DenseNet模型定义
├── densenet.pt           # 模型
├── denseNet.ipynb        
├── test_before_padding/  # 对比图片（填充前）
├── test_after_padding/   # 对比图片（填充后）
├── data/                 # MNIST数据集
└── README.md             # 项目说明文档
```

## 功能特点

### 1. 单个数字识别
- 支持上传单个数字图片
- 显示图像处理过程：原图、二值化、裁剪、缩放
- 展示预测结果和模型输出概率

### 2. 学号识别
- 支持上传包含多个数字的图片
- 自动分割并识别每个数字
- 显示完整的图像处理流程
- 展示每个分割出的数字及其识别结果

## 技术实现

### 模型架构
- 基于DenseNet架构的改进版本
- 专门针对MNIST数据集优化
- 包含多个DenseBlock和TransitionLayer
- 使用批归一化和ReLU激活函数

### 图像处理流程
1. 图像预处理
   - 灰度化
   - 二值化
   - 轮廓检测
   - 图像分割
   - 大小调整（28×28）
2. 数据增强
   - 边框填充
   - 尺寸标准化

### Web界面
- 使用Streamlit构建
- 响应式布局
- 实时预览和处理
- 直观的结果展示

## 使用说明

1. 安装依赖：
```bash
pip install torch torchvision opencv-python streamlit numpy pandas
```

2. 运行Web应用：
```bash
streamlit run front.py
```

3. 使用方法：
   - 打开浏览器访问本地服务
   - 选择"单个数字识别"或"学号识别"标签页
   - 上传图片并等待识别结果

## 注意事项

- 图片应为黑底白字或白底黑字
- 建议使用清晰的手写数字图片
- 学号识别时，数字间应有适当间距
- 支持jpg、jpeg、png格式的图片 