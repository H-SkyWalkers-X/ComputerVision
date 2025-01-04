import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])
    
    grad_x = apply_convolution(image, sobel_x)
    print(grad_x)
    grad_y = apply_convolution(image, sobel_y)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude

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
    contrast = 0
    homogeneity = 0
    energy = 0
    for i in range(256):
        for j in range(256):
            contrast += (i - j) ** 2 * glcm[i, j]
            homogeneity += glcm[i, j] / (1 + abs(i - j))
            energy += glcm[i, j] ** 2

    return np.array([contrast, homogeneity, energy])


# 二值化函数
def binarize_image(image, threshold):
    binary_image = np.zeros_like(image, dtype=np.uint8)
    binary_image[image >= threshold] = 255  # 大于等于阈值的像素设为白色
    binary_image[image < threshold] = 0     # 小于阈值的像素设为黑色
    return binary_image

# 主函数
def process_image(image_path):
 
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobel_image = sobel_filter(image)
    gray_hist = calculate_gray_histogram_manual(image)
    img=cv2.imread(image_path)
    color_hist = calculate_color_histogram(img)

    

    custom_image = custom_filter(image)
    
    
    texture_features = glcm_features(image)

    np.save('texture_features.npy', texture_features)

    return sobel_image, custom_image,color_hist,gray_hist

# 执行
image_path = './test2.jpg'  # 将此路径替换为你的图像路径



sobel_image, custom_image,color_hist,gray_hist = process_image(image_path)

# 二值化处理
threshold = 100  # 设置二值化阈值
print(sobel_image)
sobel_binary = binarize_image(sobel_image, 15)
custom_binary = binarize_image(custom_image, threshold)
cv2.imshow('Sobel Image', sobel_binary)



# 显示图像
cv2.imshow("Custom Binary Image", custom_binary)
# kernel = np.ones((3, 3), np.uint8)
# sobel_binary_enhanced = cv2.dilate(sobel_binary, kernel, iterations=1)
# custom_binary_enhanced = cv2.dilate(custom_binary, kernel, iterations=1)

plt.figure(figsize=(20, 4))
print("1111")
for i, color in enumerate(['Red', 'Green', 'Blue','Gray']):
    if i == 3:
        plt.subplot(1, 4, i+1)
        plt.plot(gray_hist)
        plt.title('Gray Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
    if i<3:
        plt.subplot(1, 4, i+1)
        plt.plot(color_hist[i])
        plt.title(f'{color} Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
print("2222")
plt.show()

# #膨胀
# kernel = np.ones((3, 3), np.uint8)
# sobel_binary_enhanced = cv2.dilate(sobel_binary, kernel, iterations=1)
# custom_binary_enhanced = cv2.dilate(custom_binary, kernel, iterations=1)
# # 显示图像
# cv2.imshow("Sobel Binary Image Enhanced", sobel_binary_enhanced)
# cv2.imshow("Custom Binary Image Enhanced", custom_binary_enhanced)


cv2.waitKey(0)
cv2.destroyAllWindows()
