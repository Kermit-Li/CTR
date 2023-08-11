import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from skimage.transform import radon

# 读取CT图像
ct_image = scipy.ndimage.imread(
    'path_to_ct_image_file', flatten=True)  # 使用你的CT图像路径

# 定义投影角度
projection_angles = np.linspace(0, 180, num=180, endpoint=False)

# 利用Siddon光线追踪法生成DRR
drr = radon(ct_image, theta=projection_angles, circle=True)

# 显示DRR图像
plt.imshow(drr, cmap='gray')
plt.title('Digitally Reconstructed Radiograph (DRR)')
plt.xlabel('Projection Angle')
plt.ylabel('Detector Position')
plt.show()
