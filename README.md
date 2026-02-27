# medicine-bag-detected-on-ui
在药品仓储管理、医院药房、零售药店及海关安检等场景中，快速准确地识别药物种类对于提升工作效率、防止误用或走私具有重要意义。本项目旨在开发一套基于 YOLOv8 的药物识别系统，能够通过图像或视频流实时检测并标注常见药物包装（如药盒、药瓶等），支持图片、视频文件及摄像头三种输入方式，并提供直观的可视化反馈。

本软件提供一个ui界面和一个权重文件，best.pt是在自建数据集上训练好的权重文件，直接在代码中调用即可。

需要的python库名：
ultralytics
opencv-python
PyQt5

注意：
1.ultralytics依赖torch，注意自己的torch，torchvision的版本。
2.最好使用conda创建新的环境，不要污染环境。

运行方法：
git项目，运行detect_gui.py
