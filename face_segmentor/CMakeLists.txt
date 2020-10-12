PROJECT(faceseg)
SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O3 -Wall -std=c++11 -DCPU_ONLY")

SET(APP_NAME test_face_seg)

# ADD_EXECUTABLE(${APP_NAME} main.cc)
ADD_LIBRARY(${APP_NAME} SHARED segmentCaffe.cpp)

FIND_PACKAGE( OpenCV REQUIRED )


INCLUDE_DIRECTORIES (
    /Users/admin/Documents/code/fswap_tool/caffe_deploy/include
	/Users/admin/Documents/code/fswap_tool/caffe_deploy/.build_release/src
#		/home/baimr/git/caffe_deploy/include
#		/home/baimr/git/caffe_deploy/.build_release/src
)

LINK_DIRECTORIES (
	/Users/admin/Documents/code/fswap_tool/caffe_deploy/.build_release/lib
#		/home/baimr/git/caffe_deploy/.build_release/lib
)

TARGET_LINK_LIBRARIES(${APP_NAME}
	${OpenCV_LIBS}
	/Users/admin/Documents/code/fswap_tool/caffe_deploy/.build_release/lib/libcaffe.so
#		/home/baimr/git/caffe_deploy/.build_release/lib/libcaffe.so
)

# ADD_SUBDIRECTORY(src)
