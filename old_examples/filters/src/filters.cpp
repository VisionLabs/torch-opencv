#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <TH/TH.h>

extern "C"
void bilateralFilter(THFloatTensor *in, THFloatTensor *out, int d, double sigmaColor, double sigmaSpace)
{
    THAssert(in->size[2] == 3);
    cv::Mat src(in->size[0], in->size[1], CV_32FC3, THFloatTensor_data(in));
    THFloatTensor_resizeAs(out, in);
    cv::Mat dst(out->size[0], out->size[1], CV_32FC3, THFloatTensor_data(out));
    cv::bilateralFilter(src, dst, d, sigmaColor, sigmaSpace);
}

extern "C"
void medianFilter(THFloatTensor *in, THFloatTensor *out, int d)
{
    THAssert(in->size[2] == 3);
    cv::Mat src(in->size[0], in->size[1], CV_32FC3, THFloatTensor_data(in));
    THFloatTensor_resizeAs(out, in);
    cv::Mat dst(out->size[0], out->size[1], CV_32FC3, THFloatTensor_data(out));
    cv::medianBlur(src, dst, d);
}

cv::VideoWriter vw;

extern "C"
void createVideoWriter(const char* name, int w, int h, float framerate)
{
    vw = cv::VideoWriter(std::string(name), CV_FOURCC('M','P','E','G'), framerate, cv::Size(w,h), true);
}

extern "C"
void addFrame(THFloatTensor *t)
{
    cv::Mat im,im2;
    if(THFloatTensor_nDimension(t) == 2)
    {
        im = cv::Mat(t->size[0], t->size[1], CV_32F, THFloatTensor_data(t));
        cv::cvtColor(im, im2, cv::COLOR_GRAY2RGB);
    }
    else
    {
        im = cv::Mat(t->size[0],t->size[1],CV_32FC3, THFloatTensor_data(t));
        im2 = im.clone();
    }
    im2.convertTo(im2, CV_8UC3, 255);
    vw.write(im2);
}

extern "C"
void releaseVideoWriter()
{
    vw.release();
}