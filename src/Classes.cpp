#include <Classes.hpp>

// FileNode

extern "C"
struct FileNodePtr FileNode_ctor() {
    return new cv::FileNode();
}

extern "C"
void FileNode_dtor(FileNodePtr ptr) {
    delete static_cast<cv::FileNode *>(ptr.ptr);
}

// FileStorage

extern "C"
struct FileStoragePtr FileStorage_ctor_default() {
    return new cv::FileStorage();
}

extern "C"
struct FileStoragePtr FileStorage_ctor(const char *source, int flags, const char *encoding) {
    return new cv::FileStorage(source, flags, encoding);
}

extern "C"
void FileStorage_dtor(FileStoragePtr ptr) {
    delete static_cast<cv::FileStorage *>(ptr.ptr);
}

extern "C"
bool FileStorage_open(FileStoragePtr ptr, const char *filename, int flags, const char *encoding) {
    return ptr->open(filename, flags, encoding);
}

extern "C"
bool FileStorage_isOpened(FileStoragePtr ptr) {
    return ptr->isOpened();
}

extern "C"
void FileStorage_release(FileStoragePtr ptr) {
    ptr->release();
}

extern "C"
const char *FileStorage_releaseAndGetString(FileStoragePtr ptr) {
    return ptr->releaseAndGetString().c_str();
}

// Algorithm

extern "C"
struct AlgorithmPtr Algorithm_ctor() {
    return new cv::Algorithm();
}

extern "C"
void Algorithm_dtor(AlgorithmPtr ptr) {
    delete static_cast<cv::Algorithm *>(ptr.ptr);
}

extern "C"
void Algorithm_clear(AlgorithmPtr ptr) {
    ptr->clear();
}

extern "C"
void Algorithm_write(AlgorithmPtr ptr, FileStoragePtr fileStorage) {
    ptr->write(*fileStorage);
}

extern "C"
void Algorithm_read(AlgorithmPtr ptr, FileNodePtr fileNode) {
    ptr->read(*fileNode);
}

extern "C"
bool Algorithm_empty(AlgorithmPtr ptr) {
    return ptr->empty();
}

extern "C"
void Algorithm_save(AlgorithmPtr ptr, const char *filename) {
    ptr->save(filename);
}

extern "C"
const char *Algorithm_getDefaultName(AlgorithmPtr ptr) {
    return ptr->getDefaultName().c_str();
}