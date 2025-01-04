import cv2

# 读取图像
img = cv2.imread(r"W:\Codes\VsCode\VsCode_Storage\mnist\test_images\1_7.jpg")
print(img.shape)
# 调整图像大小，这里将图像调整为360x480
# img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA)

# # 保存调整后的图像
# cv2.imwrite("./test2.jpg", img)