# CTR
from ct to get drr and train a model to register 2d X-ray images

Please add my **qq:235636351** to facilitate further communication, which is not very convenient through the intermediary.

# 配置安装
- cmake = 3.27.2
- ITK = 5.3.0

# 安装教程
- cmake 安装或者更新 (https://zhuanlan.zhihu.com/p/513871916)
- ITK 配置 (https://blog.csdn.net/chen499093551/article/details/91528309)

# 功能示意

## CT to DRR
- 文件夹: ctdrr
    - eg：
        - 查看命令参数: ./ctdrr/ctdrr -h
        - CT转DRR: ./ctdrr/ctdrr -size 1024 1024 -rx 90 -ry 0 -rz 0 -t 10 10 -10 -o "./drr/2.png" "./nii/1.nii"
        - 上例中rx,ry,rz为旋转角度，-t后的3个参数为位移参数。-o后的为输出图像，最后文件为读取的三维文件。