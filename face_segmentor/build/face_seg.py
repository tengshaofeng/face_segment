from ctypes import *
import time
import cv2
import numpy as np
foo = CDLL('libtest_face_seg.so')
# tic = time.time()
# for i in range(10):
#     foo.predict_seg_fname(c_char_p('../data/model_01_seq00.jpg'), c_char_p('res.jpg'))
#
# res=(time.time()-tic)/10
# print(res)

class MemImg(Structure):
    _fields_ = [
        ('width', c_int),
        ('height', c_int),
        ('type', c_int),
        ('data', c_char_p),
        ('mask_data', c_char_p),
    ]
def gen_mem_img(part,  img_type=cv2.CV_8UC3, mask=None):
    if part is not None:
        if mask is not None:
            mask_data = cast(mask.ctypes.data_as(POINTER(c_ubyte*(mask.shape[0] * mask.shape[1] * mask.shape[2]))), c_char_p)
        else:
            mask_data = c_char_p(0)
        part_img = MemImg(
            width = part.shape[1],
            height = part.shape[0],
            #stride = part.shape[1] * part.shape[2],
            type = img_type,
            data = cast(part.ctypes.data_as(POINTER(c_ubyte*(part.shape[0] * part.shape[1] * part.shape[2]))), c_char_p),
            mask_data = mask_data
        )
    else:
        part_img = MemImg(0)
    return part_img



im = cv2.imread('../data/model_01_seq00.jpg')
res_mat = np.ndarray((224, 224, 3), dtype=np.uint8)
res_mask_mat = np.ndarray((224, 224), dtype=np.uint8)
out_img_mem_size = 224 * 224 * 3
out_mask_mem_size = 224 * 224
out_img = MemImg(
    data = cast(res_mat.ctypes.data_as(POINTER(c_ubyte*out_img_mem_size)), c_char_p),
    mask_data = cast(res_mask_mat.ctypes.data_as(POINTER(c_ubyte*out_mask_mem_size)), c_char_p)
)
in_img = gen_mem_img(im)
tic = time.time()
for i in range(10):
    foo.predict_seg(byref(in_img), byref(out_img))
avgtime = (time.time()-tic)/10
print(avgtime)

cv2.imwrite('tbq1.jpg', res_mat)
cv2.imwrite('tbq2.jpg', res_mask_mat)
