# face_segment
## how to run?
This is a implement of face segmentation using caffe.

You can call run a demo to do face segmentation by:
#
cd face_segmentor

sudo mkdir /usr/local/lib/python2.7/dist-packages/face_rebuild_py/face_rebuild_lib/dll_and_datv2/models/face_seg_fcn8s

cp data/HGNet_S2_deploy.prototxt /usr/local/lib/python2.7/dist-packages/face_rebuild_py/face_rebuild_lib/dll_and_datv2/models/face_seg_fcn8s

cp data/HGNet_S2_train_new2_iter_120000.caffemodel  /usr/local/lib/python2.7/dist-packages/face_rebuild_py/face_rebuild_lib/dll_and_datv2/models/face_seg_fcn8s

python face_seg.py
#
## result
input:  ![image](https://github.com/tengshaofeng/face_segment/blob/master/face_segmentor/data/model_01_seq00.jpg)

output: ![image](https://github.com/tengshaofeng/face_segment/blob/master/face_segmentor/res.png)

## The following is dose not matter:
### generate build/libtest_face_seg.so

cd face_segmentor

mkdir build

cmake ..

make -j4

