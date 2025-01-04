import cv2
import numpy as np
import math

'''
先膨胀后腐蚀，参数调大，虽然结果看不到了白色线，但是仍能够在差异化后的图像中看到
与Canny相比，差异化后的图像明显少了一堆噪声
'''



# 读取图像
image = cv2.imread('./test2.jpg')

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)

# 应用高斯模糊以减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow('blurred', blurred)

#! 使用Canny边缘检测得到处理的二值图像
edges = cv2.Canny(blurred, 50, 150)
cv2.imshow('edges', edges)



# cv2.imshow('edges2', binary_image)
kernel = np.ones((3, 3), np.uint8)
binary_enhanced = cv2.dilate(edges, kernel, iterations=10)

cv2.imshow('dilate', binary_enhanced)

binary_enhanced = cv2.erode(binary_enhanced, kernel, iterations=20)

cv2.imshow('dilate and erode', binary_enhanced)

difference = cv2.subtract(edges, binary_enhanced)
cv2.imshow('difference', difference)

#!  先腐蚀后膨胀不可行
# binary_enhanced = cv2.erode(edges, kernel, iterations=5)
# cv2.imshow('erode', binary_enhanced)
# binary_enhanced = cv2.dilate(binary_enhanced, kernel, iterations=10)
# cv2.imshow('erode and dilate', binary_enhanced)


height, width = difference.shape
# mask = np.zeros_like(difference)
# polygon = np.array([[(100, height), (width - 50, height), (width - 50, height-height //3), (100, height-height //3)]], dtype=np.int32)
# cv2.fillPoly(mask, polygon, 255)
# roi = cv2.bitwise_and(difference, mask)

# 霍夫变换检测直线
r_max = int(math.sqrt(width**2 + height**2))
accumulator = np.zeros((2 * r_max, 180), dtype=int)


for y in range(height):
    for x in range(width):
        if difference[y, x] != 0:  # 只对ROI内的边缘点进行霍夫变换
            for theta in range(0, 180):
                r = int(x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta)))
                r_idx = r + r_max
                accumulator[r_idx, theta] += 1

threshold = 100  # 阈值，过滤掉噪声
for r_idx in range(accumulator.shape[0]):
    for theta in range(accumulator.shape[1]):
        if accumulator[r_idx, theta] > threshold:
            r = r_idx - r_max
            theta = np.deg2rad(theta)
            x1 = int(r * np.cos(theta) - 1000 * np.sin(theta))
            y1 = int(r * np.sin(theta) + 1000 * np.cos(theta))
            x2 = int(r * np.cos(theta) + 1000 * np.sin(theta))
            y2 = int(r * np.sin(theta) - 1000 * np.cos(theta))

            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 显示结果
cv2.imshow('Lane Detection with Optimized Hough Transform', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
