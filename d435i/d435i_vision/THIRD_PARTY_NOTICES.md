# Third-Party Notices

## smart-logistics-vision 色环检测算法

`d435i_vision/algorithm/ring_detector.py` 中的色环预处理与 Hough 圆检测逻辑复制并改编自
本机来源文件：

```text
/home/book/Desktop/smart-logistics-vision/src/logistics_vision/detectors/circles.py
```

上游项目：[xijun536/smart-logistics-vision](https://github.com/xijun536/smart-logistics-vision)
（本次改编所参考的本地版本为提交
[`a2b4fb3e4cf5b12788dd0b627238dc6180924769`](https://github.com/xijun536/smart-logistics-vision/tree/a2b4fb3e4cf5b12788dd0b627238dc6180924769)）。
其算法思路来源于 CSDN 原创文章
[《工创赛2025-智能物流搬运塔吊方案视觉开源（2分15秒）》](https://blog.csdn.net/2303_80277100/article/details/150211965)。

上游项目与原文声明采用
[Creative Commons Attribution-ShareAlike 4.0 International（CC BY-SA 4.0）](https://creativecommons.org/licenses/by-sa/4.0/)，
[完整法律文本](https://creativecommons.org/licenses/by-sa/4.0/legalcode)可在线查看。
本项目中来自该来源的改编内容继续按 CC BY-SA 4.0 提供。

`cv3` 保留了上游的图像预处理流程和 `cv2.HOUGH_GRADIENT_ALT` 检测思路，包括
形态学处理、梯度/阈值中间图、画面边界过滤及同心候选合并。主要语义改动是：
上游结果契约要求恰好三个圆，而 `cv3` 返回零个或任意数量的有效圆，并按 X 坐标
从左到右排序。任何正数色环都视为成功；只有恰好三个色环时才生成中间圆中心和
左右圆 Y 差字段。
