# 调用人脸分割网络的例子

## 简介

因为[FaceSeg类](http://gitlab.taofen8.com/baimr/fswap/blob/master/include/face_seg.hpp#L10) 的2个调用，和前面的关键点以及对齐过程的耦合过强。

所以做了一个demo，输入为图片，输出为mask。

（建议输入为对齐的人脸图像。）
