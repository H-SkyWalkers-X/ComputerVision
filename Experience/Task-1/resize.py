import cv2

# 读取图像
img = cv2.imread("./t1.jpg")
print(img.shape)
# 调整图像大小，这里将图像调整为360x480
img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA)
kernel_size=(15, 15)
sigma=1
blurred_image = cv2.GaussianBlur(img, kernel_size, sigma)
cv2.imshow("image",blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# # 保存调整后的图像
# cv2.imwrite("./test2.jpg", img)