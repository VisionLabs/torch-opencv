#include <opencv2/core.hpp>
#include <Common.hpp>

// This is cv::Ptr with all fields made public
// TODO I hope a safer solution to be here one day
template <typename T>
struct PublicPtr
{
public:
    cv::detail::PtrOwner* owner;
    T* stored;
};

// increfs a cv::Ptr and returns the pointer
template <typename T>
T *rescueObjectFromPtr(cv::Ptr<T> ptr) {
    PublicPtr<T> *publicPtr = reinterpret_cast<PublicPtr<T> *>(&ptr);
    publicPtr->owner->incRef();
    return ptr.get();
}

// FileNode

struct FileNodePtr {
    void *ptr;

    inline cv::FileNode * operator->() { return static_cast<cv::FileNode *>(ptr); }
    inline FileNodePtr(cv::FileNode *ptr) { this->ptr = ptr; }
    inline cv::FileNode & operator*() { return *static_cast<cv::FileNode *>(this->ptr); }
};

extern "C"
struct FileNodePtr FileNode_ctor();

extern "C"
void FileNode_dtor(FileNodePtr ptr);

// FileStorage

struct FileStoragePtr {
    void *ptr;

    inline cv::FileStorage * operator->() { return static_cast<cv::FileStorage *>(ptr); }
    inline FileStoragePtr(cv::FileStorage *ptr) { this->ptr = ptr; }
    inline cv::FileStorage & operator*() { return *static_cast<cv::FileStorage *>(this->ptr); }
};

extern "C"
struct FileStoragePtr FileStorage_ctor_default();

extern "C"
struct FileStoragePtr FileStorage_ctor(const char *source, int flags, const char *encoding);

extern "C"
void FileStorage_dtor(FileStoragePtr ptr);

extern "C"
bool FileStorage_open(FileStoragePtr ptr, const char *filename, int flags, const char *encoding);

extern "C"
bool FileStorage_isOpened(FileStoragePtr ptr);

extern "C"
void FileStorage_release(FileStoragePtr ptr);

extern "C"
const char *FileStorage_releaseAndGetString(FileStoragePtr ptr);

// Algorithm

struct AlgorithmPtr {
    void *ptr;

    inline cv::Algorithm * operator->() { return static_cast<cv::Algorithm *>(ptr); }
    inline AlgorithmPtr(cv::Algorithm *ptr) { this->ptr = ptr; }
};

extern "C"
struct AlgorithmPtr Algorithm_ctor();

extern "C"
void Algorithm_dtor(AlgorithmPtr ptr);

extern "C"
void Algorithm_clear(AlgorithmPtr ptr);

extern "C"
void Algorithm_write(AlgorithmPtr ptr, FileStoragePtr fileStorage);

extern "C"
void Algorithm_read(AlgorithmPtr ptr, FileNodePtr fileNode);

extern "C"
bool Algorithm_empty(AlgorithmPtr ptr);

extern "C"
void Algorithm_save(AlgorithmPtr ptr, const char *filename);

extern "C"
const char *Algorithm_getDefaultName(AlgorithmPtr ptr);