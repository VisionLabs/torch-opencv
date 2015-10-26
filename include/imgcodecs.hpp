#include <Common.hpp>
#include <opencv2/imgcodecs.hpp>

extern "C"
struct TensorWrapper imread(const char *filename, int flags);

extern "C"
struct TensorArrayPlusBool imreadmulti(const char *filename, int flags);

extern "C"
bool imwrite(const char *filename, struct TensorWrapper img, struct TensorWrapper params);

extern "C"
struct TensorWrapper imdecode(struct TensorWrapper buf, int flags);
