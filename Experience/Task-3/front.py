import streamlit as st
import cv2
import torch
import numpy as np
from model import DenseNetMNIST
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd   
st.set_page_config(page_title="数字识别系统", layout="wide")
st.title("数字识别系统")                    
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DenseNetMNIST(num_classes=10).to(device)
    model.load_state_dict(torch.load('./densenet.pt'))
    model.eval()
    return model, device

model, device = load_model()
tab1, tab2 = st.tabs(["单个数字识别", "学号识别"])

# 单个数字识别
with tab1:
    st.header("单个数字识别")
    uploaded_file = st.file_uploader("选择一张包含单个数字的图片", type=['jpg', 'jpeg', 'png'], key="single")
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # 转为灰度图并二值化
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 获取最大的外接矩形
            x, y, w, h = cv2.boundingRect(contours[0])
            # 裁剪图像到最小矩形
            cropped_image = binary_image[y:y + h, x:x + w]
            # 调整尺寸到28x28
            resized_image = cv2.resize(cropped_image, (28, 28), interpolation=cv2.INTER_AREA)
            # 进行预测
            image_tensor = transform(resized_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                predicted_label = predicted.item()
            # 输出
            st.markdown("""
                <style>
                    .result-box {
                        padding: 20px;
                        background-color: #f0f2f6;
                        border-radius: 10px;
                        text-align: center;
                        margin-bottom: 20px;
                    }
                    /* 设置数据表格的样式 */
                    .stDataFrame {
                        font-size: 1.5rem !important;
                    }
                    .stDataFrame td, .stDataFrame th {
                        font-size: 1.5rem !important;
                        padding: 10px !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="result-box">
                    <h2>预测结果: {predicted_label}</h2>
                </div>
            """, unsafe_allow_html=True)
            output_data = output.cpu().numpy()[0]
            output_df = pd.DataFrame([output_data], columns=[f'数字{i}' for i in range(10)])
            st.write("模型输出:")
            st.dataframe(output_df, hide_index=True)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("原始图片")
                st.image(image, channels="BGR")
            with col2:
                st.subheader("二值化图片")
                st.image(binary_image)
            
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("裁剪后的图片")
                st.image(cropped_image)
            with col4:
                st.subheader("调整大小后的图片(28x28)")
                st.image(resized_image)

# 学号识别
with tab2:
    st.header("学号识别")
    uploaded_file = st.file_uploader("选择一张包含学号的图片", type=['jpg', 'jpeg', 'png'], key="multi")
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (480, 360), interpolation=cv2.INTER_AREA)
        
        # 转为灰度图并进行二值化
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY_INV)
        
        # 增强处理
        kernel = np.ones((3, 3), np.uint8)
        binary_enhanced = cv2.dilate(binary_image, kernel, iterations=1)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(binary_enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        # 存储每个分割出的数字
        digit_images = []
        contour_image = image.copy()
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # 在原图上画出矩形框
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            digit_image = binary_image[y:y + h, x:x + w]
            # 添加黑色边框
            digit_image = cv2.copyMakeBorder(
                digit_image,
                5,  # 上边填充5像素
                5,  # 下边填充5像素
                9,  # 左边填充9像素
                9,  # 右边填充9像素
                cv2.BORDER_CONSTANT,
                value=0
            )
            digit_images.append(digit_image)
        # 对每个数字进行预测
        predicted_labels = []
        if digit_images:
            for digit_image in digit_images:
                digit_image_resized = cv2.resize(digit_image, (28, 28), interpolation=cv2.INTER_AREA)
                digit_image_tensor = transform(digit_image_resized).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(digit_image_tensor)
                    _, predicted = torch.max(output, 1)
                    predicted_labels.append(predicted.item())
        # 显示结果
        if predicted_labels:
            st.markdown("""
                <style>
                    .result-box {
                        padding: 20px;
                        background-color: #f0f2f6;
                        border-radius: 10px;
                        text-align: center;
                        margin-bottom: 20px;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="result-box">
                    <h2>识别结果: {''.join(map(str, predicted_labels))}</h2>
                </div>
            """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("原始图片")
            st.image(image, channels="BGR")
        with col2:
            st.subheader("二值化图片")
            st.image(binary_image)
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("增强处理后的图片")
            st.image(binary_enhanced)
        with col4:
            st.subheader("检测到的数字位置")
            st.image(contour_image, channels="BGR")
        if digit_images:
            st.subheader(f"检测到 {len(digit_images)} 个数字")
            digits_per_row = 5
            num_rows = (len(digit_images) + digits_per_row - 1) // digits_per_row
            
            for row in range(num_rows):
                cols = st.columns(digits_per_row)
                start_idx = row * digits_per_row
                end_idx = min(start_idx + digits_per_row, len(digit_images))
                
                for idx in range(start_idx, end_idx):
                    with cols[idx - start_idx]:
                        st.image(digit_images[idx], caption=f"第{idx+1}个数字，识别为：{predicted_labels[idx]}")
