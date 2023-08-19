# CTR
from ct to get drr and train a model to register 2d X-ray images

编程代写，闲鱼搜索PickQStudio，选择优质代码，提供优质项目服务，加**qq:235636351**，了解更多。

# 编译环境，语言和工具
- Linux
- Python
- C++
- Vscode

# 配置安装
- cmake = 3.27.2
- ITK = 5.3.0

# 安装教程
- cmake 安装或者更新 (https://zhuanlan.zhihu.com/p/513871916)
- ITK 配置 (https://blog.csdn.net/chen499093551/article/details/91528309)

# 文件结构
- dataset (文件夹): 存放原始dcm数据
- nii (文件夹): 存放3维ct数据
- drr (文件夹): 存放2维生成DRR数据
- ctdrr (文件夹): ct 转 drr 可执行文件
    - ctdrr/ctdrr 
- dcm_to_nii.ipynb: 含有从原始到生成drr和设置(x,y,z,$\theta$,$\alpha$,$\beta$)的流程示例代码
- train.ipynb: 含有训练drr神经网络模型的流程示例代码
    - model.py: 模型代码文件
    - utils.py: 加载数据，训练，验证等代码模块
---------生成文件------------
- metadata.csv: drr目录下的drr图像基础信息
- loss.jpg: 训练过程中训练和验证损失曲线
- model.pt: 训练完成后保存的模型文件，加载利用`model = torch.load("model.pt")`

# 功能示意
## CT to DRR
- 可执行文件1: ctdrr
    - eg：
        - 查看命令参数(参看更多参数功能): ./ctdrr/ctdrr -h
        - CT转DRR: ./ctdrr/ctdrr -size 1024 1024 -rx 90 -ry 0 -rz 0 -t 10 10 -10 -o "./drr/2.png" "./nii/1.nii"
        - 上例中rx,ry,rz为旋转角度，-t后的3个参数为位移参数。-o后的为输出图像，最后文件为读取的三维文件。

## 其他文件与训练网络模型有关
- 数据处理流程: dcm_to_nii.ipynb
    - eg: 利用1个CT序列文件生成DRR（1000）张全过程，具体见notebook
- 网络训练流程: train.ipynb
    - eg: 加载数据并划分数据集，训练网络，保存模型，具体见notebook
