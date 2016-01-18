#include <Common.hpp>
#include <opencv2/calib3d.hpp>

extern "C"
double calibrateCamera(
	struct TensorArray objectPoints, struct TensorArray imagePoints,
	struct SizeWrapper imageSize, struct TensorWrapper cameraMatrix,
	struct TensorWrapper distCoeffs, struct TensorArray rvecs,
	struct TensorArray tvecs, int flags, struct TermCriteriaWrapper criteria);

extern "C"
struct TensorWrapper calibrationMatrixValues(
	struct TensorWrapper cameraMatrix,
	struct SizeWrapper imageSize,
	double apertureWidth, double apertureHeight);

extern "C"
void composeRT(
	struct TensorWrapper rvec1, struct TensorWrapper tvec1, struct TensorWrapper rvec2,
	struct TensorWrapper tvec2, struct TensorWrapper rvec3, struct TensorWrapper tvec3,
	struct TensorWrapper dr3dr1, struct TensorWrapper dr3dt1, struct TensorWrapper dr3dr2,
	struct TensorWrapper dr3dt2, struct TensorWrapper dt3dr1, struct TensorWrapper dt3dt1,
	struct TensorWrapper dt3dr2, struct TensorWrapper dt3dt2);

extern "C"
struct TensorWrapper computeCorrespondEpilines(
	struct TensorWrapper points, int whichImage, struct TensorWrapper F);

extern "C"
struct TensorWrapper convertPointsFromHomogeneous(
	struct TensorWrapper src);

extern "C"
struct TensorWrapper convertPointsHomogeneous(
	struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct TensorWrapper convertPointsToHomogeneous(
	struct TensorWrapper src);

extern "C"
struct TensorArray correctMatches(
	struct TensorWrapper F, struct TensorWrapper points1,
	struct TensorWrapper points2);

extern "C"
struct TensorArray decomposeEssentialMat(
	struct TensorWrapper E);

extern "C"
struct TensorArrayPlusInt decomposeHomographyMat(
	struct TensorWrapper H, struct TensorWrapper K);

extern "C"
struct TensorArray decomposeProjectionMatrix(
	struct TensorWrapper projMatrix, struct TensorWrapper rotMatrixX,
	struct TensorWrapper rotMatrixY, struct TensorWrapper rotMatrixZ,
	struct TensorWrapper eulerAngles);

extern "C"
void drawChessboardCorners(
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper corners, bool patternWasFound);

extern "C"
struct TensorArrayPlusInt estimateAffine3D(
	struct TensorWrapper src, struct TensorWrapper dst,
	double ransacThreshold, double confidence);

extern "C"
void filterSpeckles(
	struct TensorWrapper img, double newVal, int maxSpeckleSize,
	double maxDiff, struct TensorWrapper buf);
 
extern "C"
void find4QuadCornerSubpix(
	struct TensorWrapper img, struct TensorWrapper corners,
	struct SizeWrapper region_size);

extern "C"
struct TensorWrapper findChessboardCorners(
	struct TensorWrapper image, struct SizeWrapper patternSize, int flags);

//TODO const Ptr<FeatureDetector>& blobDetector = SimpleBlobDetector::create()
extern "C"
struct TensorPlusBool findCirclesGrid(
	struct TensorWrapper image, struct SizeWrapper patternSize, int flags);

extern "C"
struct TensorWrapper findEssentialMat(
	struct TensorWrapper points1, struct TensorWrapper points2,
	double focal, struct Point2dWrapper pp, int method, double prob,
	double threshold, struct TensorWrapper mask);

extern "C"
struct TensorWrapper findFundamentalMat(
	struct TensorWrapper points1, struct TensorWrapper points2, int method,
	double param1, double param2, struct TensorWrapper mask);

extern "C"
struct TensorWrapper findFundamentalMat2(
	struct TensorWrapper points1, struct TensorWrapper points2,
	struct TensorWrapper mask, int method, double param1, double param2);

extern "C"
struct TensorWrapper findHomography(
	struct TensorWrapper srcPoints, struct TensorWrapper dstPoints,
	int method, double ransacReprojThreshold, struct TensorWrapper mask,
	const int maxIters, const double confidence);

extern "C"
struct TensorWrapper findHomography2(
	struct TensorWrapper srcPoints, struct TensorWrapper dstPoints,
	struct TensorWrapper mask, int method, double ransacReprojThreshold);


