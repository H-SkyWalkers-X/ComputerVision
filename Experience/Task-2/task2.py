import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# 读取图像
image = cv2.imread('./test2.jpg')
# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 应用高斯模糊以减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# 使用Canny边缘检测得到处理的二值图像
edges = cv2.Canny(blurred, 50, 150)
# 膨胀处理
kernel = np.ones((3, 3), np.uint8)
after_dilate = cv2.dilate(edges, kernel, iterations=10)
# 腐蚀处理
after_erode = cv2.erode(after_dilate, kernel, iterations=20)
# 差异化处理
difference = cv2.subtract(edges, after_erode)


# kernel2 = np.array([[2,  2,  2],
#                    [ -1, -1, -1],
#                    [2,  2,  2]], dtype=np.float32)

# difference = cv2.filter2D(difference, -1, kernel2)

# _, difference = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)





# 获取图像的尺寸
height, width = difference.shape
r_max = int(math.sqrt(width ** 2 + height ** 2))  # 最大半径值
accumulator = np.zeros((2 * r_max, 180), dtype=int)  # 初始化霍夫变换的累加器
for y in range(height):
    for x in range(width):
        if difference[y, x] != 0:  # 只对边缘点进行霍夫变换
            for theta in range(0, 180):  # 角度范围从0到179
                r = int(x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta)))  # 计算r值
                r_idx = r + r_max  # 处理r值的偏移
                accumulator[r_idx, theta] += 1  # 增加累加器的投票值
threshold = 100
lines = []
for r_idx in range(accumulator.shape[0]):
    for theta in range(accumulator.shape[1]):
        if accumulator[r_idx, theta] > threshold:  # 如果该位置的投票数大于阈值
            r = r_idx - r_max  # 还原r值
            theta = np.deg2rad(theta)  # 转换为弧度
            # 计算直线的两个端点
            x1 = int(r * np.cos(theta) - 1000 * np.sin(theta))
            y1 = int(r * np.sin(theta) + 1000 * np.cos(theta))
            x2 = int(r * np.cos(theta) + 1000 * np.sin(theta))
            y2 = int(r * np.sin(theta) - 1000 * np.cos(theta))
            # 存储检测到的直线
            lines.append((x1, y1, x2, y2))

# 合并重复的直线（采用一个容忍度阈值）
def merge_lines(lines, angle_threshold=20, distance_threshold=30):
    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        line_found = False
        for i, (mx1, my1, mx2, my2) in enumerate(merged_lines):
            # 计算两个线段之间的角度差和距离差
            angle1 = math.atan2(y2 - y1, x2 - x1)
            angle2 = math.atan2(my2 - my1, mx2 - mx1)
            angle_diff = abs(angle1 - angle2) * 180 / np.pi
            distance = abs((x2 - x1) * (my1 - y1) - (y2 - y1) * (mx1 - x1)) / math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if angle_diff < angle_threshold and distance < distance_threshold:
                # 合并直线
                merged_lines[i] = (min(x1, mx1), min(y1, my1), max(x2, mx2), max(y2, my2))
                line_found = True
                break
        if not line_found:
            merged_lines.append(line)
    return merged_lines
# 合并直线
merged_lines = merge_lines(lines)
# 在原图上绘制合并后的直线
for line in merged_lines:
    x1, y1, x2, y2 = line
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 将图像从BGR转换为RGB，以便matplotlib正确显示
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 定义一个函数来执行霍夫变换和直线检测
def detect_lines(input_image, threshold=100):
    height, width = input_image.shape
    r_max = int(math.sqrt(width ** 2 + height ** 2))
    accumulator = np.zeros((2 * r_max, 180), dtype=int)
    
    for y in range(height):
        for x in range(width):
            if input_image[y, x] != 0:
                for theta in range(0, 180):
                    r = int(x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta)))
                    r_idx = r + r_max
                    accumulator[r_idx, theta] += 1
    
    lines = []
    for r_idx in range(accumulator.shape[0]):
        for theta in range(accumulator.shape[1]):
            if accumulator[r_idx, theta] > threshold:
                r = r_idx - r_max
                theta = np.deg2rad(theta)
                x1 = int(r * np.cos(theta) - 1000 * np.sin(theta))
                y1 = int(r * np.sin(theta) + 1000 * np.cos(theta))
                x2 = int(r * np.cos(theta) + 1000 * np.sin(theta))
                y2 = int(r * np.sin(theta) - 1000 * np.cos(theta))
                lines.append((x1, y1, x2, y2))
    
    return lines

# 输出三种不同的情况
# 1. 完整处理（包含差异化和形态学操作）
lines_full = detect_lines(difference)
image_full = image.copy()
merged_lines_full = merge_lines(lines_full)
for line in merged_lines_full:
    x1, y1, x2, y2 = line
    cv2.line(image_full, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 2. 不使用差异化（只有形态学操作）
lines_no_diff = detect_lines(after_erode)
image_no_diff = image.copy()
merged_lines_no_diff = merge_lines(lines_no_diff)
for line in merged_lines_no_diff:
    x1, y1, x2, y2 = line
    cv2.line(image_no_diff, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 3. 只使用边缘检测（不使用形态学操作和差异化）
lines_only_edges = detect_lines(edges)
image_only_edges = image.copy()
merged_lines_only_edges = merge_lines(lines_only_edges)
for line in merged_lines_only_edges:
    x1, y1, x2, y2 = line
    cv2.line(image_only_edges, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 转换为RGB以便显示
image_full_rgb = cv2.cvtColor(image_full, cv2.COLOR_BGR2RGB)
image_no_diff_rgb = cv2.cvtColor(image_no_diff, cv2.COLOR_BGR2RGB)
image_only_edges_rgb = cv2.cvtColor(image_only_edges, cv2.COLOR_BGR2RGB)

# 创建输出目录
output_dir = './output_images/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 显示处理步骤（5张图）
plt.figure(figsize=(15, 10))
plt.subplot(231)
plt.imshow(edges, cmap='gray')
plt.title('(a) Edge Detection (Canny)')
plt.axis('off')
plt.subplot(232)
plt.imshow(after_dilate, cmap='gray')
plt.title('(b) Dilated Image')
plt.axis('off')
plt.subplot(233)
plt.imshow(after_erode, cmap='gray')
plt.title('(c) Dilate and Erode')
plt.axis('off')
plt.subplot(234)
plt.imshow(difference, cmap='gray')
plt.title('(d) Differenced Image')
plt.axis('off')
plt.subplot(235)
plt.imshow(image_rgb)
plt.title('(e) Final Image with Hough Lines')
plt.axis('off')
plt.subplot(236)
plt.axis('off')
plt.tight_layout()
plt.savefig(output_dir + 'process_results.png', dpi=300, bbox_inches='tight')
plt.show()
# 2. 显示不使用差异化的结果（单张图）
plt.figure(figsize=(8, 6))
plt.imshow(image_no_diff_rgb)
plt.title('Without Difference Operation')
plt.axis('off')
plt.tight_layout()
plt.savefig(output_dir + 'no_difference_result.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 显示只使用边缘检测的结果（单张图）
plt.figure(figsize=(8, 6))
plt.imshow(image_only_edges_rgb)
plt.title('Only Edge Detection')
plt.axis('off')
plt.tight_layout()
plt.savefig(output_dir + 'only_edges_result.png', dpi=300, bbox_inches='tight')
plt.show()

cv2.imwrite(output_dir + 'edges.png', edges)
cv2.imwrite(output_dir + 'dilated.png', after_dilate)
cv2.imwrite(output_dir + 'eroded.png', after_erode)
cv2.imwrite(output_dir + 'difference.png', difference)
cv2.imwrite(output_dir + 'final_result.png', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(output_dir + 'no_difference.png', cv2.cvtColor(image_no_diff_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(output_dir + 'only_edges.png', cv2.cvtColor(image_only_edges_rgb, cv2.COLOR_RGB2BGR))