#include <iostream>
#include <string>
#include <ciso646>
#include <vector>
#include <memory>
#include <cstring>

#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>

typedef unsigned char uchar;
typedef void* pCvMat;


struct MemImg {
    int width;
    int height;
    int type;             // CV_8UC3 == 16 or CV_8UC4 == 24
    unsigned char* data;  // BGR or BGRA
    unsigned char* mask_data;
};

cv::Mat plot_mask_on_im(const cv::Mat& im, const cv::Mat& mask) {
    const int COLS = im.cols;
    assert(im.cols==mask.cols and im.rows==mask.rows);
    cv::Mat plot_im = im.clone();
    for(int h=0; h<im.rows; ++h) {
        for(int w=0; w<im.cols; ++w) {
            if(mask.data[h*COLS+w]==0) {
                plot_im.data[(h*COLS+w)*3] = 0;
                plot_im.data[(h*COLS+w)*3+1] = 255;
                plot_im.data[(h*COLS+w)*3+2] = 0;
            }
        }
    }
    return plot_im;
}


class SegmentLib
{

    public:
        SegmentLib();
        void predict(const MemImg* in_img, MemImg* out_img);
        void predict_fname(const char* input_im, const char* output_im);

    private:
        std::shared_ptr<caffe::Net<float>> ptr_seg_net;
        const cv::Vec3f m_mean_val{ 103.939f, 116.779f, 123.68f };
        const float m_scale_val = 1.0f / 128.0f;
        const float m_prob_thres = 0.5f;

};

SegmentLib::SegmentLib(){
    // 载入网络
    // build face segmentation nn
    char deploy_path [] = "/usr/local/lib/python2.7/dist-packages/face_rebuild_py/face_rebuild_lib/"
                          "dll_and_data_v2/models/face_seg_fcn8s/HGNet_S2_deploy.prototxt";
    char weight_path [] = "/usr/local/lib/python2.7/dist-packages/face_rebuild_py/face_rebuild_lib/"
                          "dll_and_data_v2/models/face_seg_fcn8s/HGNet_S2_train_new2_iter_120000.caffemodel";
    std::cout << deploy_path << std::endl;

    auto a = caffe::TEST;
    std::cout << a << std::endl;

    this->ptr_seg_net = std::shared_ptr<caffe::Net<float>>(new caffe::Net<float>(deploy_path, caffe::TEST));
    this->ptr_seg_net->CopyTrainedLayersFrom(weight_path);

}

void SegmentLib::predict(const MemImg* in_img, MemImg* out_img){
// 限定单张输入 并resize到 224 x 224
    const int input_size_len = 224;
    std::cout << in_img << std::endl;
//    cv::Mat cur_aligned_img = cv::imread(input_im);
    //cv::Mat cur_aligned_img = input_im
    //std::cout << cur_aligned_img << std::endl;
    cv::Mat in_mat(in_img->height, in_img->width, in_img->type, const_cast<unsigned char*>(in_img->data));
    assert(in_mat.channels()==3 or in_mat.channels() == 4);
    if(in_mat.channels() == 4)
        cv::cvtColor(in_mat, in_mat, CV_BGRA2BGR);
    if(in_mat.rows != input_size_len or in_mat.cols != input_size_len) {
        std::cout << "resize input image to " << input_size_len << " x " << input_size_len << std::endl;
        cv::resize(in_mat, in_mat, cv::Size(input_size_len, input_size_len));
    }

    // 预处理 ( 按照FCN_FaceSeg::doFaceSeg()改写 )

    cv::Mat aligned_tensor = in_mat.clone();
    in_mat.convertTo(aligned_tensor, CV_32FC3);        // 转化dtype
    for (int h = 0; h < aligned_tensor.rows; ++h) {
        for (int w = 0; w < aligned_tensor.cols; ++w) {
            aligned_tensor.at<cv::Vec3f>(h, w) =
                    (aligned_tensor.at<cv::Vec3f>(h, w) - m_mean_val) * m_scale_val;
        }
    }



    // 前向
    float* input_data = ptr_seg_net->input_blobs()[0]->mutable_cpu_data();

    std::vector<cv::Mat> input_channels(3);
    cv::split(aligned_tensor, input_channels);
    float* cur_p = input_data;
    for(int i=0; i<3; ++i) {        // todo 用 emplace_back 写split后零拷贝的版本
        std::memcpy(cur_p, (float*)input_channels[i].data, input_size_len*input_size_len*sizeof(float));
        cur_p += input_size_len*input_size_len;
    }

    ptr_seg_net->Forward();

    // 后处理 ( (按照FCN_FaceSeg::postProcessSegNetOut改写) )

    caffe::Blob<float>* output_layer = ptr_seg_net->output_blobs()[0];          //
    cv::Mat prob_mat(input_size_len, input_size_len, CV_32FC1, (void*)output_layer->cpu_data());
    cv::Mat out_seg_mask = cv::Mat::zeros(input_size_len, input_size_len, CV_8UC1);
    uchar *out_seg_mask_data = (uchar*)out_seg_mask.data;
    float *prob_mat_data = (float*)prob_mat.data;

    for (int i = 0; i < input_size_len*input_size_len; ++i) {
        if (prob_mat_data[i] > m_prob_thres + 0.1f)
            out_seg_mask_data[i] = 255;
    }
//    返回分割后的图片和mask
    cv::Mat out_seg = plot_mask_on_im(in_mat, out_seg_mask);
    out_img->width = out_seg.cols;
    out_img->height = out_seg.rows;
    out_img->type = out_seg.type();
    memcpy(out_img->data, out_seg.data, out_seg.step*out_seg.rows);
    memcpy(out_img->mask_data, out_seg_mask.data, out_seg_mask.step*out_seg_mask.rows);
    // cv::imwrite(output_im.c_str(), out_seg_mask);
    cv::imwrite("tbq.png", out_seg);
}

void SegmentLib::predict_fname(const char* input_im, const char* output_im){
// 限定单张输入 并resize到 224 x 224
    const int input_size_len = 224;
    std::cout << input_im << std::endl;
    cv::Mat cur_aligned_img = cv::imread(input_im);
    //cv::Mat cur_aligned_img = input_im
    //std::cout << cur_aligned_img << std::endl;
    assert(cur_aligned_img.channels()==3 or cur_aligned_img.channels() == 4);
    if(cur_aligned_img.channels() == 4)
        cv::cvtColor(cur_aligned_img, cur_aligned_img, CV_BGRA2BGR);
    if(cur_aligned_img.rows != input_size_len or cur_aligned_img.cols != input_size_len) {
        std::cout << "resize input image to " << input_size_len << " x " << input_size_len << std::endl;
        cv::resize(cur_aligned_img, cur_aligned_img, cv::Size(input_size_len, input_size_len));
    }

    // 预处理 ( 按照FCN_FaceSeg::doFaceSeg()改写 )

    cv::Mat aligned_tensor = cur_aligned_img.clone();
    cur_aligned_img.convertTo(aligned_tensor, CV_32FC3);        // 转化dtype
    for (int h = 0; h < aligned_tensor.rows; ++h) {
        for (int w = 0; w < aligned_tensor.cols; ++w) {
            aligned_tensor.at<cv::Vec3f>(h, w) =
                    (aligned_tensor.at<cv::Vec3f>(h, w) - m_mean_val) * m_scale_val;
        }
    }



    // 前向
    float* input_data = ptr_seg_net->input_blobs()[0]->mutable_cpu_data();

    std::vector<cv::Mat> input_channels(3);
    cv::split(aligned_tensor, input_channels);
    float* cur_p = input_data;
    for(int i=0; i<3; ++i) {        // todo 用 emplace_back 写split后零拷贝的版本
        std::memcpy(cur_p, (float*)input_channels[i].data, input_size_len*input_size_len*sizeof(float));
        cur_p += input_size_len*input_size_len;
    }

    ptr_seg_net->Forward();

    // 后处理 ( (按照FCN_FaceSeg::postProcessSegNetOut改写) )

    caffe::Blob<float>* output_layer = ptr_seg_net->output_blobs()[0];          //
    cv::Mat prob_mat(input_size_len, input_size_len, CV_32FC1, (void*)output_layer->cpu_data());
    cv::Mat out_seg_mask = cv::Mat::zeros(input_size_len, input_size_len, CV_8UC1);
    uchar *out_seg_mask_data = (uchar*)out_seg_mask.data;
    float *prob_mat_data = (float*)prob_mat.data;

    for (int i = 0; i < input_size_len*input_size_len; ++i) {
        if (prob_mat_data[i] > m_prob_thres + 0.1f)
            out_seg_mask_data[i] = 255;
    }

    // cv::imwrite(output_im.c_str(), out_seg_mask);
    cv::imwrite(output_im, plot_mask_on_im(cur_aligned_img, out_seg_mask));

}


SegmentLib obj;
extern "C"{
    void predict_seg(const MemImg* in_img, MemImg* out_img){
        obj.predict(in_img, out_img);
    }
    void predict_seg_fname(const char* input_im, const char* output_im){
        obj.predict_fname(input_im, output_im);
    }
}

