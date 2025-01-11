import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# 卷积函数
def apply_convolution(image, kernel):
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    output_image = np.zeros_like(image)

    # 卷积操作，只在有效区域进行
    for i in range(pad_height, img_height - pad_height):
        for j in range(pad_width, img_width - pad_width):
            region = image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            output_image[i, j] = np.sum(region * kernel)

    return output_image

# Sobel滤波
def sobel_filter(image):
    # 高斯
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    grad_x = apply_convolution(image, sobel_x)
    grad_y = apply_convolution(image, sobel_y)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    
    # 归一化梯度幅值到0-1范围
    magnitude = magnitude / magnitude.max()
    # 非极大值抑制
    nms = non_maximum_suppression(magnitude, direction)
    # 双阈值检测
    strong_edges, weak_edges = double_threshold(nms, low_ratio=0.1, high_ratio=0.3)
    
    # 滞后边缘跟踪
    final_edges = hysteresis(strong_edges, weak_edges)
    
    return magnitude, final_edges

# 给定卷积核滤波
def custom_filter(image):
    kernel = np.array([[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]])
    return apply_convolution(image, kernel)

# 计算灰度直方图
def calculate_gray_histogram_manual(image):
    hist = [0] * 256  # 初始化 256 个灰度级的计数器
    img_height, img_width = image.shape

    for i in range(img_height):
        for j in range(img_width):
            pixel_value = image[i, j]
            hist[pixel_value] += 1  # 增加对应灰度级的计数

    return hist
def calculate_color_histogram(image):
    color_hist = []
    for i in range(3):  # 对于RGB每个通道
        hist = np.histogram(image[:, :, i], bins=256, range=(0, 256))[0]
        color_hist.append(hist)
    return color_hist



# 纹理特征提取：计算灰度共生矩阵（GLCM）
def glcm_features(image):
    glcm = np.zeros((256, 256), dtype=np.int32)
    height, width = image.shape
    for i in range(height-1):
        for j in range(width-1):
            glcm[image[i, j], image[i+1, j+1]] += 1
            
    # 归一化 GLCM
    glcm_normalized = glcm / np.sum(glcm)
    
    contrast = 0
    homogeneity = 0
    energy = 0
    for i in range(256):
        for j in range(256):
            contrast += (i - j) ** 2 * glcm[i, j]
            homogeneity += glcm[i, j] / (1 + abs(i - j))
            energy += glcm[i, j] ** 2

    return glcm, np.array([contrast, homogeneity, energy])


# 二值化函数
def binarize_image(image, threshold):
    binary_image = np.zeros_like(image, dtype=np.uint8)
    binary_image[image >= threshold] = 255  # 大于等于阈值的像素设为白色
    binary_image[image < threshold] = 0     # 小于阈值的像素设为黑色
    return binary_image

# 主函数
def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 获取 Sobel 滤波的幅值图和边缘图
    sobel_magnitude, sobel_edges = sobel_filter(image)
    gray_hist = calculate_gray_histogram_manual(image)
    img = cv2.imread(image_path)
    color_hist = calculate_color_histogram(img)
    
    custom_image = custom_filter(image)
    
    # 获取 GLCM 和纹理特征
    glcm, texture_features = glcm_features(image)
    
    np.save('texture_features.npy', texture_features)

    return sobel_magnitude, sobel_edges, custom_image, color_hist, gray_hist, glcm

# 非极大值抑制
def non_maximum_suppression(magnitude, direction):
    height, width = magnitude.shape
    result = np.zeros_like(magnitude)
    
    # 将角度转换到0-180度
    direction = np.rad2deg(direction) % 180
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            # 获取相邻的两个像素点
            angle = direction[i,j]
            
            # 四个主要方向：0°, 45°, 90°, 135°
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):  # 0度
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elif (22.5 <= angle < 67.5):  # 45度
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            elif (67.5 <= angle < 112.5):  # 90度
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            else:  # 135度
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            
            # 如果当前点是局部最大值，则保留
            if magnitude[i,j] >= max(neighbors):
                result[i,j] = magnitude[i,j]
    
    return result

# 双阈值检测
def double_threshold(img, low_ratio=0.1, high_ratio=0.3):
    # 计算高低阈值
    high_threshold = np.percentile(img[img > 0], 100 * (1 - high_ratio))
    low_threshold = high_threshold * low_ratio
    
    strong_edges = (img >= high_threshold)
    weak_edges = (img >= low_threshold) & (img < high_threshold)
    
    return strong_edges, weak_edges

# 滞后边缘跟踪
def hysteresis(strong_edges, weak_edges):
    height, width = strong_edges.shape
    result = np.copy(strong_edges)
    
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]
    
    # 遍历所有弱边缘点
    while True:
        before = np.sum(result)
        for i in range(1, height-1):
            for j in range(1, width-1):
                if weak_edges[i,j] and not result[i,j]:
                    # 检查8邻域是否有强边缘点
                    if np.any(result[i-1:i+2, j-1:j+2]):
                        result[i,j] = True
        
        if np.sum(result) == before:
            break
    
    return result

# 修改主程序显示部分
def display_results(image_path):
    sobel_magnitude, sobel_edges, custom_image, color_hist, gray_hist, glcm = process_image(image_path)
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    output_dir = './output_images/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(12, 10)) 
    
    # 原始图像
    plt.subplot(221)
    plt.imshow(original_image, cmap='gray')
    plt.title('(a) Original Image')
    plt.axis('off')
    
    # Sobel 
    plt.subplot(222)
    plt.imshow(sobel_magnitude, cmap='gray')
    plt.title('(b) Sobel Magnitude')
    plt.axis('off')
    
    # Sobel 
    plt.subplot(223)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title('(c) Sobel Edges')
    plt.axis('off')
    
    # 自定义滤波结果
    plt.subplot(224)
    plt.imshow(binarize_image(custom_image, 15), cmap='gray')
    plt.title('(d) Custom Filter Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir + 'edge_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 单独保存每个处理结果
    cv2.imwrite(output_dir + 'original.png', original_image)
    cv2.imwrite(output_dir + 'sobel_magnitude.png', (sobel_magnitude * 255).astype(np.uint8))
    cv2.imwrite(output_dir + 'sobel_edges.png', (sobel_edges * 255).astype(np.uint8))
    cv2.imwrite(output_dir + 'custom_filter.png', binarize_image(custom_image, 15))
    
    # 直方图
    plt.figure(figsize=(12, 10))  # 调整为正方形图像
    colors = ['Red', 'Green', 'Blue', 'Gray']
    titles = ['(a) Red', '(b) Green', '(c) Blue', '(d) Gray']  # 新增标题列表
    
    for i, (color, title) in enumerate(zip(colors, titles)):
        plt.subplot(2, 2, i+1)  # 改为2x2布局
        if color == 'Gray':
            plt.plot(gray_hist, color='gray')
        else:
            plt.plot(color_hist[colors.index(color)], color=color.lower())
        plt.title(f'{title} Channel Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)  # 添加网格线使图像更清晰
    
    plt.tight_layout()
    plt.savefig(output_dir + 'histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # GLCM
    plt.figure(figsize=(8, 6))
    plt.imshow(np.log1p(glcm), cmap='hot')
    plt.colorbar()
    plt.title('Gray Level Co-occurrence Matrix (Log Scale)')
    plt.xlabel('Gray Level i')
    plt.ylabel('Gray Level j')
    plt.tight_layout()
    plt.savefig(output_dir + 'glcm.png', dpi=300, bbox_inches='tight')
    plt.show()

    np.save(output_dir + 'color_histograms.npy', color_hist) #对比度
    np.save(output_dir + 'gray_histogram.npy', gray_hist) #同质性
    np.save(output_dir + 'glcm_matrix.npy', glcm) #能量

image_path = './test2.jpg'
display_results(image_path)
