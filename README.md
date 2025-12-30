本项目是基于cmake的CUDAC++代码库，用来记录 cuda 的学习代码。

# 一、章节介绍：
0. 学习《CUDA编程：基础与实践》一书
1. 学习 CUDA reduce，来自bilibili UP https://space.bilibili.com/218427631/lists/4695308?type=series
2. 学习 CUDA sgemm，来自bilibili UP https
://space.bilibili.com/218427631/lists/4695308?type=series


# 二、调试/运行环境：
Windows10 64位
Nvidia GTX 1660Ti
Nvidia Driver 560.94
CUDA Toolkit 12.6.2
CMake 4.2.1
vscode 1.106.1
vscode插件：
    C/C++
    C/C++ Extension Pack
    CMake
    Cmake Tools
    Nsight Visual Studio Code Edition
WSL2


# 三、注意事项：
1. 如果不调试，请注释掉debug配置，改配置会显著影响运行耗时。（开debug配置，你的优化结果会让你怀疑人生~.~）
2. debug需要将 build生成的二进制文件绝对路径 填写为 ./vscode/launch.json 文件的 configurations.program 的值。
3. TBD

