#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/objdetect.hpp>

struct BaseCascadeClassifierPtr {
    void *ptr;

    inline cv::BaseCascadeClassifier * operator->() { return static_cast<cv::BaseCascadeClassifier *>(ptr); }
    inline BaseCascadeClassifierPtr(cv::BaseCascadeClassifier *ptr) { this->ptr = ptr; }
    inline cv::BaseCascadeClassifier & operator*() { return *static_cast<cv::BaseCascadeClassifier *>(this->ptr); }
};

struct CascadeClassifierPtr {
    void *ptr;

    inline cv::CascadeClassifier * operator->() { return static_cast<cv::CascadeClassifier *>(ptr); }
    inline CascadeClassifierPtr(cv::CascadeClassifier *ptr) { this->ptr = ptr; }
    inline cv::CascadeClassifier & operator*() { return *static_cast<cv::CascadeClassifier *>(this->ptr); }
};

