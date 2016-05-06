#include <Common.hpp>
#include <Classes.hpp>
#include <opencv2/imgproc.hpp>

extern "C"
struct TensorWrapper getGaussianKernel(int ksize, double sigma, int ktype);

extern "C"
struct TensorArray getDerivKernels(
        int dx, int dy, int ksize, struct TensorWrapper kx,
        struct TensorWrapper ky, bool normalize, int ktype);

extern "C"
struct TensorWrapper getGaborKernel(struct SizeWrapper ksize, double sigma, double theta,
                                               double lambd, double gamma, double psi, int ktype);

extern "C"
struct TensorWrapper getStructuringElement(int shape, struct SizeWrapper ksize,
                                                      struct PointWrapper anchor);

extern "C"
struct TensorWrapper medianBlur(struct TensorWrapper src, int ksize, struct TensorWrapper dst);

extern "C"
struct TensorWrapper GaussianBlur(struct TensorWrapper src, struct SizeWrapper ksize,
                                  double sigmaX, struct TensorWrapper dst,
                                  double sigmaY, int borderType);

extern "C"
struct TensorWrapper bilateralFilter(struct TensorWrapper src, int d,
                                     double sigmaColor, double sigmaSpace,
                                     struct TensorWrapper dst, int borderType);

extern "C"
struct TensorWrapper boxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct SizeWrapper ksize, struct PointWrapper anchor,
        bool normalize, int borderType);

extern "C"
struct TensorWrapper sqrBoxFilter(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct SizeWrapper ksize, struct PointWrapper anchor,
        bool normalize, int borderType);

extern "C"
struct TensorWrapper blur(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper ksize, struct PointWrapper anchor, int borderType);

extern "C"
struct TensorWrapper filter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        double delta, int borderType);

extern "C"
struct TensorWrapper sepFilter2D(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        struct TensorWrapper kernelX,struct TensorWrapper kernelY,
        struct PointWrapper anchor, double delta, int borderType);

extern "C"
struct TensorWrapper Sobel(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int dx, int dy, int ksize, double scale, double delta, int borderType);

extern "C"
struct TensorWrapper Scharr(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int dx, int dy, double scale, double delta, int borderType);

extern "C"
struct TensorWrapper Laplacian(
        struct TensorWrapper src, struct TensorWrapper dst, int ddepth,
        int ksize, double scale, double delta, int borderType);

extern "C"
struct TensorWrapper Canny(
        struct TensorWrapper image, struct TensorWrapper edges,
        double threshold1, double threshold2, int apertureSize, bool L2gradient);

extern "C"
struct TensorWrapper cornerMinEigenVal(
        struct TensorWrapper src, struct TensorWrapper dst,
        int blockSize, int ksize, int borderType);

extern "C"
struct TensorWrapper cornerHarris(
        struct TensorWrapper src, struct TensorWrapper dst, int blockSize,
        int ksize, double k, int borderType);

extern "C"
struct TensorWrapper cornerEigenValsAndVecs(
        struct TensorWrapper src, struct TensorWrapper dst,
        int blockSize, int ksize, int borderType);

extern "C"
struct TensorWrapper preCornerDetect(
        struct TensorWrapper src, struct TensorWrapper dst, int ksize, int borderType);

extern "C"
struct TensorWrapper HoughLines(
        struct TensorWrapper image,
        double rho, double theta, int threshold, double srn, double stn,
        double min_theta, double max_theta);

extern "C"
struct TensorWrapper HoughLinesP(
        struct TensorWrapper image, double rho,
        double theta, int threshold, double minLineLength, double maxLineGap);

extern "C"
struct TensorWrapper HoughCircles(
        struct TensorWrapper image,
        int method, double dp, double minDist, double param1, double param2,
        int minRadius, int maxRadius);

extern "C"
void cornerSubPix(
        struct TensorWrapper image, struct TensorWrapper corners,
        struct SizeWrapper winSize, struct SizeWrapper zeroZone,
        struct TermCriteriaWrapper criteria);

extern "C"
struct TensorWrapper goodFeaturesToTrack(
        struct TensorWrapper image,
        int maxCorners, double qualityLevel, double minDistance,
        struct TensorWrapper mask, int blockSize, bool useHarrisDetector, double k);

extern "C"
struct TensorWrapper erode(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue);

extern "C"
struct TensorWrapper dilate(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue);

extern "C"
struct TensorWrapper morphologyEx(
        struct TensorWrapper src, struct TensorWrapper dst,
        int op, struct TensorWrapper kernel, struct PointWrapper anchor,
        int iterations, int borderType, struct ScalarWrapper borderValue);

extern "C"
struct TensorWrapper resize(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dsize, double fx, double fy,
        int interpolation);

extern "C"
struct TensorWrapper warpAffine(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue);

extern "C"
struct TensorWrapper warpPerspective(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper M, struct SizeWrapper dsize,
        int flags, int borderMode, struct ScalarWrapper borderValue);

extern "C"
struct TensorWrapper remap(
        struct TensorWrapper src, struct TensorWrapper map1,
        struct TensorWrapper map2, int interpolation, struct TensorWrapper dst,
        int borderMode, struct ScalarWrapper borderValue);

extern "C"
struct TensorArray convertMaps(
        struct TensorWrapper map1, struct TensorWrapper map2,
        struct TensorWrapper dstmap1, struct TensorWrapper dstmap2,
        int dstmap1type, bool nninterpolation);

extern "C"
struct TensorWrapper getRotationMatrix2D(
        struct Point2fWrapper center, double angle, double scale);

extern "C"
struct TensorWrapper invertAffineTransform(
        struct TensorWrapper M, struct TensorWrapper iM);

extern "C"
struct TensorWrapper getPerspectiveTransform(
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct TensorWrapper getAffineTransform(
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct TensorWrapper getRectSubPix(
        struct TensorWrapper image, struct SizeWrapper patchSize,
        struct Point2fWrapper center, struct TensorWrapper patch, int patchType);

extern "C"
struct TensorWrapper logPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct Point2fWrapper center, double M, int flags);

extern "C"
struct TensorWrapper linearPolar(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct Point2fWrapper center, double maxRadius, int flags);

extern "C"
struct TensorWrapper integral(
        struct TensorWrapper src, struct TensorWrapper sum, int sdepth);

extern "C"
struct TensorArray integralN(
        struct TensorWrapper src, struct TensorArray sums, int sdepth, int sqdepth);

extern "C"
void accumulate(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper mask);

extern "C"
void accumulateSquare(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper mask);

extern "C"
void accumulateProduct(
        struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper dst, struct TensorWrapper mask);

extern "C"
void accumulateWeighted(
        struct TensorWrapper src, struct TensorWrapper dst,
        double alpha, struct TensorWrapper mask);

extern "C"
struct Vec3dWrapper phaseCorrelate(
        struct TensorWrapper src1, struct TensorWrapper src2,
        struct TensorWrapper window);

extern "C"
struct TensorWrapper createHanningWindow(
        struct TensorWrapper dst, struct SizeWrapper winSize, int type);

extern "C"
struct TensorPlusDouble threshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double thresh, double maxval, int type);

extern "C"
struct TensorWrapper adaptiveThreshold(
        struct TensorWrapper src, struct TensorWrapper dst,
        double maxValue, int adaptiveMethod, int thresholdType,
        int blockSize, double C);

extern "C"
struct TensorWrapper pyrDown(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dstSize, int borderType);

extern "C"
struct TensorWrapper pyrUp(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct SizeWrapper dstSize, int borderType);

extern "C"
struct TensorArray buildPyramid(
        struct TensorWrapper src, struct TensorArray dst,
        int maxlevel, int borderType);

extern "C"
struct TensorWrapper undistort(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper newCameraMatrix);

extern "C"
struct TensorArray initUndistortRectifyMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper newCameraMatrix,
        struct SizeWrapper size, int m1type,
        struct TensorArray maps);

extern "C"
struct TensorArrayPlusFloat initWideAngleProjMap(
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct SizeWrapper imageSize, int destImageWidth,
        int m1type, struct TensorArray maps,
        int projType, double alpha);

extern "C"
struct TensorWrapper getDefaultNewCameraMatrix(
        struct TensorWrapper cameraMatrix, struct SizeWrapper imgsize, bool centerPrincipalPoint);

extern "C"
struct TensorWrapper undistortPoints(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
        struct TensorWrapper R, struct TensorWrapper P);

extern "C"
struct TensorWrapper calcHist(
        struct TensorArray images,
        struct TensorWrapper channels, struct TensorWrapper mask,
        struct TensorWrapper hist, int dims, struct TensorWrapper histSize,
        struct TensorWrapper ranges, bool uniform, bool accumulate);

extern "C"
struct TensorWrapper calcBackProject(
        struct TensorArray images, int nimages,
        struct TensorWrapper channels, struct TensorWrapper hist,
        struct TensorWrapper backProject, struct TensorWrapper ranges,
        double scale, bool uniform);

extern "C"
double compareHist(
        struct TensorWrapper H1, struct TensorWrapper H2, int method);

extern "C"
struct TensorWrapper equalizeHist(
        struct TensorWrapper src, struct TensorWrapper dst);

extern "C"
struct TensorPlusFloat EMD(
        struct TensorWrapper signature1, struct TensorWrapper signature2,
        int distType, struct TensorWrapper cost,
        struct TensorWrapper lowerBound, struct TensorWrapper flow);

extern "C"
void watershed(
        struct TensorWrapper image, struct TensorWrapper markers);

extern "C"
struct TensorWrapper pyrMeanShiftFiltering(
        struct TensorWrapper src, struct TensorWrapper dst,
        double sp, double sr, int maxLevel, struct TermCriteriaWrapper termcrit);

extern "C"
struct TensorArray grabCut(
        struct TensorWrapper img, struct TensorWrapper mask,
        struct RectWrapper rect, struct TensorWrapper bgdModel,
        struct TensorWrapper fgdModel, int iterCount, int mode);

extern "C"
struct TensorWrapper distanceTransform(
        struct TensorWrapper src, struct TensorWrapper dst,
        int distanceType, int maskSize, int dstType);

extern "C"
struct TensorArray distanceTransformWithLabels(
        struct TensorWrapper src, struct TensorWrapper dst,
        struct TensorWrapper labels, int distanceType, int maskSize,
        int labelType);

extern "C"
struct RectPlusInt floodFill(
        struct TensorWrapper image, struct TensorWrapper mask,
        struct PointWrapper seedPoint, struct ScalarWrapper newVal,
        struct ScalarWrapper loDiff, struct ScalarWrapper upDiff, int flags);

extern "C"
struct TensorWrapper cvtColor(
        struct TensorWrapper src, struct TensorWrapper dst, int code, int dstCn);

extern "C"
struct TensorWrapper demosaicing(
        struct TensorWrapper _src, struct TensorWrapper _dst, int code, int dcn);

extern "C"
struct MomentsWrapper moments(
        struct TensorWrapper array, bool binaryImage);

extern "C"
struct TensorWrapper HuMoments(
        struct MomentsWrapper m);

extern "C"
struct TensorWrapper matchTemplate(
        struct TensorWrapper image, struct TensorWrapper templ, struct TensorWrapper result, int method, struct TensorWrapper mask);

extern "C"
struct TensorPlusInt connectedComponents(
        struct TensorWrapper image, struct TensorWrapper labels, int connectivity, int ltype);

extern "C"
struct TensorArrayPlusInt connectedComponentsWithStats(
        struct TensorWrapper image, struct TensorArray outputTensors, int connectivity, int ltype);

extern "C"
struct TensorArray findContours(
        struct TensorWrapper image, bool withHierarchy, struct TensorWrapper hierarchy, int mode, int method, struct PointWrapper offset);

extern "C"
struct TensorWrapper approxPolyDP(
        struct TensorWrapper curve, struct TensorWrapper approxCurve, double epsilon, bool closed);

extern "C"
double arcLength(
        struct TensorWrapper curve, bool closed);

extern "C"
struct RectWrapper boundingRect(
        struct TensorWrapper points);

extern "C"
double contourArea(
        struct TensorWrapper contour, bool oriented);

extern "C"
struct RotatedRectWrapper minAreaRect(
        struct TensorWrapper points);

extern "C"
struct TensorWrapper boxPoints(
        struct RotatedRectWrapper box, struct TensorWrapper points);

extern "C"
struct Vec3fWrapper minEnclosingCircle(
        struct TensorWrapper points, struct Point2fWrapper center, float radius);

extern "C"
struct TensorPlusDouble minEnclosingTriangle(
        struct TensorWrapper points, struct TensorWrapper triangle);

extern "C"
double matchShapes(
        struct TensorWrapper contour1, struct TensorWrapper contour2, int method, double parameter);

extern "C"
struct TensorWrapper convexHull(
        struct TensorWrapper points, struct TensorWrapper hull,
        bool clockwise, bool returnPoints);

extern "C"
struct TensorWrapper convexityDefects(
        struct TensorWrapper contour, struct TensorWrapper convexhull,
        struct TensorWrapper convexityDefects);

extern "C"
bool isContourConvex(
        struct TensorWrapper contour);

extern "C"
struct TensorPlusFloat intersectConvexConvex(
        struct TensorWrapper _p1, struct TensorWrapper _p2,
        struct TensorWrapper _p12, bool handleNested);

extern "C"
struct RotatedRectWrapper fitEllipse(
        struct TensorWrapper points);

extern "C"
struct TensorWrapper fitLine(
        struct TensorWrapper points, struct TensorWrapper line, int distType,
        double param, double reps, double aeps);

extern "C"
double pointPolygonTest(
        struct TensorWrapper contour, struct Point2fWrapper pt, bool measureDist);

extern "C"
struct TensorWrapper rotatedRectangleIntersection(
        struct RotatedRectWrapper rect1, struct RotatedRectWrapper rect2);

extern "C"
struct TensorWrapper blendLinear(
        struct TensorWrapper src1, struct TensorWrapper src2, struct TensorWrapper weights1, struct TensorWrapper weights2, struct TensorWrapper dst);

extern "C"
struct TensorWrapper applyColorMap(
        struct TensorWrapper src, struct TensorWrapper dst, int colormap);

extern "C"
void line(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2, struct ScalarWrapper color, int thickness, int lineType, int shift);

extern "C"
void arrowedLine(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2, struct ScalarWrapper color, int thickness, int line_type, int shift, double tipLength);

extern "C"
void rectangle(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2, struct ScalarWrapper color, int thickness, int lineType, int shift);

extern "C"
void rectangle2(
        struct TensorWrapper img, struct RectWrapper rec, struct ScalarWrapper color, int thickness, int lineType, int shift);

extern "C"
void circle(
        struct TensorWrapper img, struct PointWrapper center, int radius, struct ScalarWrapper color, int thickness, int lineType, int shift);

extern "C"
void ellipse(
        struct TensorWrapper img, struct PointWrapper center, struct SizeWrapper axes, double angle, double startAngle, double endAngle, struct ScalarWrapper color, int thickness, int lineType, int shift);

extern "C"
void ellipseFromRect(
        struct TensorWrapper img, struct RotatedRectWrapper box, struct ScalarWrapper color, int thickness, int lineType);

extern "C"
void fillConvexPoly(
        struct TensorWrapper img, struct TensorWrapper points, struct ScalarWrapper color, int lineType, int shift);

extern "C"
void fillPoly(
        struct TensorWrapper img, struct TensorArray pts, struct ScalarWrapper color, int lineType, int shift, struct PointWrapper offset);

extern "C"
void polylines(
        struct TensorWrapper img, struct TensorArray pts, bool isClosed, struct ScalarWrapper color, int thickness, int lineType, int shift);

extern "C"
void drawContours(
        struct TensorWrapper image, struct TensorArray contours, int contourIdx, struct ScalarWrapper color, int thickness, int lineType, struct TensorWrapper hierarchy, int maxLevel, struct PointWrapper offset);

extern "C"
struct ScalarPlusBool clipLineSize(
        struct SizeWrapper imgSize, struct PointWrapper pt1, struct PointWrapper pt2);

extern "C"
struct ScalarPlusBool clipLineRect(
        struct RectWrapper imgRect, struct PointWrapper pt1, struct PointWrapper pt2);

extern "C"
struct TensorWrapper ellipse2Poly(
        struct PointWrapper center, struct SizeWrapper axes, int angle, int arcStart, int arcEnd, int delta);

extern "C"
void putText(
        struct TensorWrapper img, const char *text, struct PointWrapper org, int fontFace, double fontScale, struct ScalarWrapper color, int thickness, int lineType, bool bottomLeftOrigin);

extern "C"
struct SizePlusInt getTextSize(
        const char *text, int fontFace, double fontScale, int thickness);

struct GeneralizedHoughPtr {
    void *ptr;

    inline cv::GeneralizedHough * operator->() { return static_cast<cv::GeneralizedHough *>(ptr); }
    inline GeneralizedHoughPtr(cv::GeneralizedHough *ptr) { this->ptr = ptr; }
};

struct GeneralizedHoughBallardPtr {
    void *ptr;

    inline cv::GeneralizedHoughBallard * operator->() { return static_cast<cv::GeneralizedHoughBallard *>(ptr); }
    inline GeneralizedHoughBallardPtr(cv::GeneralizedHoughBallard *ptr) { this->ptr = ptr; }
};

struct GeneralizedHoughGuilPtr {
    void *ptr;

    inline cv::GeneralizedHoughGuil * operator->() { return static_cast<cv::GeneralizedHoughGuil *>(ptr); }
    inline GeneralizedHoughGuilPtr(cv::GeneralizedHoughGuil *ptr) { this->ptr = ptr; }
};

struct CLAHEPtr {
    void *ptr;

    inline cv::CLAHE * operator->() { return static_cast<cv::CLAHE *>(ptr); }
    inline CLAHEPtr(cv::CLAHE *ptr) { this->ptr = ptr; }
};

struct LineSegmentDetectorPtr {
    void *ptr;

    inline cv::LineSegmentDetector * operator->() { return static_cast<cv::LineSegmentDetector *>(ptr); }
    inline LineSegmentDetectorPtr(cv::LineSegmentDetector *ptr) { this->ptr = ptr; }
};

struct Subdiv2DPtr {
    void *ptr;

    inline cv::Subdiv2D * operator->() { return static_cast<cv::Subdiv2D *>(ptr); }
    inline Subdiv2DPtr(cv::Subdiv2D *ptr) { this->ptr = ptr; }
};

struct LineIteratorPtr {
    void *ptr;

    inline cv::LineIterator * operator->() { return static_cast<cv::LineIterator *>(ptr); }
    inline LineIteratorPtr(cv::LineIterator *ptr) { this->ptr = ptr; }
};

extern "C"
void GeneralizedHough_setTemplate(
        GeneralizedHoughPtr ptr, struct TensorWrapper templ, struct PointWrapper templCenter);

extern "C"
void GeneralizedHough_setTemplate_edges(
        GeneralizedHoughPtr ptr, struct TensorWrapper edges, struct TensorWrapper dx,
        struct TensorWrapper dy, struct PointWrapper templCenter);

extern "C"
struct TensorArray GeneralizedHough_detect(
        GeneralizedHoughPtr ptr, struct TensorWrapper image, struct TensorWrapper positions, bool votes);

extern "C"
struct TensorArray GeneralizedHough_detect_edges(
        GeneralizedHoughPtr ptr, struct TensorWrapper edges, struct TensorWrapper dx,
        struct TensorWrapper dy, struct TensorWrapper positions, bool votes);

extern "C"
void GeneralizedHough_setCannyLowThresh(GeneralizedHoughPtr ptr, int cannyLowThresh);

extern "C"
int GeneralizedHough_getCannyLowThresh(GeneralizedHoughPtr ptr);

extern "C"
void GeneralizedHough_setCannyHighThresh(GeneralizedHoughPtr ptr, int cannyHighThresh);

extern "C"
int GeneralizedHough_getCannyHighThresh(GeneralizedHoughPtr ptr);

extern "C"
void GeneralizedHough_setMinDist(GeneralizedHoughPtr ptr, double MinDist);

extern "C"
double GeneralizedHough_getMinDist(GeneralizedHoughPtr ptr);

extern "C"
void GeneralizedHough_setDp(GeneralizedHoughPtr ptr, double Dp);

extern "C"
double GeneralizedHough_getDp(GeneralizedHoughPtr ptr);

extern "C"
void GeneralizedHough_setMaxBufferSize(GeneralizedHoughPtr ptr, int MaxBufferSize);

extern "C"
int GeneralizedHough_getMaxBufferSize(GeneralizedHoughPtr ptr);

extern "C"
struct GeneralizedHoughBallardPtr GeneralizedHoughBallard_ctor();

extern "C"
void GeneralizedHoughBallard_setLevels(GeneralizedHoughBallardPtr ptr, double Levels);

extern "C"
double GeneralizedHoughBallard_getLevels(GeneralizedHoughBallardPtr ptr);

extern "C"
void GeneralizedHoughBallard_setVotesThreshold(GeneralizedHoughBallardPtr ptr, double votesThreshold);

extern "C"
double GeneralizedHoughBallard_getVotesThreshold(GeneralizedHoughBallardPtr ptr);

extern "C"
struct GeneralizedHoughGuilPtr GeneralizedHoughGuil_ctor();

extern "C"
void GeneralizedHoughGuil_setLevels(GeneralizedHoughGuilPtr ptr, int levels);

extern "C"
int GeneralizedHoughGuil_getLevels(GeneralizedHoughGuilPtr ptr);

extern "C"
void GeneralizedHoughGuil_setAngleEpsilon(GeneralizedHoughGuilPtr ptr, double AngleEpsilon);

extern "C"
double GeneralizedHoughGuil_getAngleEpsilon(GeneralizedHoughGuilPtr ptr);

extern "C"
void GeneralizedHoughGuil_setMinAngle(GeneralizedHoughGuilPtr ptr, double MinAngle);

extern "C"
double GeneralizedHoughGuil_getMinAngle(GeneralizedHoughGuilPtr ptr);

extern "C"
void GeneralizedHoughGuil_setMaxAngle(GeneralizedHoughGuilPtr ptr, double MaxAngle);

extern "C"
double GeneralizedHoughGuil_getMaxAngle(GeneralizedHoughGuilPtr ptr);

extern "C"
void GeneralizedHoughGuil_setAngleStep(GeneralizedHoughGuilPtr ptr, double AngleStep);

extern "C"
double GeneralizedHoughGuil_getAngleStep(GeneralizedHoughGuilPtr ptr);

extern "C"
void GeneralizedHoughGuil_setAngleThresh(GeneralizedHoughGuilPtr ptr, int AngleThresh);

extern "C"
int GeneralizedHoughGuil_getAngleThresh(GeneralizedHoughGuilPtr ptr);

extern "C"
void GeneralizedHoughGuil_setMinScale(GeneralizedHoughGuilPtr ptr, double MinScale);

extern "C"
double GeneralizedHoughGuil_getMinScale(GeneralizedHoughGuilPtr ptr);

extern "C"
void GeneralizedHoughGuil_setMaxScale(GeneralizedHoughGuilPtr ptr, double MaxScale);

extern "C"
double GeneralizedHoughGuil_getMaxScale(GeneralizedHoughGuilPtr ptr);

extern "C"
void GeneralizedHoughGuil_setScaleStep(GeneralizedHoughGuilPtr ptr, double ScaleStep);

extern "C"
double GeneralizedHoughGuil_getScaleStep(GeneralizedHoughGuilPtr ptr);

extern "C"
void GeneralizedHoughGuil_setScaleThresh(GeneralizedHoughGuilPtr ptr, int ScaleThresh);

extern "C"
int GeneralizedHoughGuil_getScaleThresh(GeneralizedHoughGuilPtr ptr);

extern "C"
void GeneralizedHoughGuil_setPosThresh(GeneralizedHoughGuilPtr ptr, int PosThresh);

extern "C"
int GeneralizedHoughGuil_getPosThresh(GeneralizedHoughGuilPtr ptr);

extern "C"
struct CLAHEPtr CLAHE_ctor();

extern "C"
void CLAHE_setClipLimit(CLAHEPtr ptr, double ClipLimit);

extern "C"
double CLAHE_getClipLimit(CLAHEPtr ptr);

extern "C"
void CLAHE_setTilesGridSize(CLAHEPtr ptr, struct SizeWrapper TilesGridSize);

extern "C"
struct SizeWrapper CLAHE_getTilesGridSize(CLAHEPtr ptr);

extern "C"
void CLAHE_collectGarbage(CLAHEPtr ptr);

extern "C"
struct LineSegmentDetectorPtr LineSegmentDetector_ctor(
        int refine, double scale, double sigma_scale, double quant,
        double ang_th, double log_eps, double density_th, int n_bins);

extern "C"
struct TensorArray LineSegmentDetector_detect(
        struct LineSegmentDetectorPtr ptr, struct TensorWrapper image,
        struct TensorWrapper lines, bool width, bool prec, bool nfa);

extern "C"
struct TensorWrapper LineSegmentDetector_drawSegments(
        struct LineSegmentDetectorPtr ptr, struct TensorWrapper image, struct TensorWrapper lines);

extern "C"
extern "C"
int LineSegmentDetector_compareSegments(struct LineSegmentDetectorPtr ptr, struct SizeWrapper size, struct TensorWrapper lines1,
                    struct TensorWrapper lines2, struct TensorWrapper image);

extern "C"
struct Subdiv2DPtr Subdiv2D_ctor_default();

extern "C"
struct Subdiv2DPtr Subdiv2D_ctor(struct RectWrapper rect);

extern "C"
void Subdiv2D_dtor(struct Subdiv2DPtr ptr);

extern "C"
void Subdiv2D_initDelaunay(struct Subdiv2DPtr ptr, struct RectWrapper rect);

extern "C"
int Subdiv2D_insert(struct Subdiv2DPtr ptr, struct Point2fWrapper pt);

extern "C"
void Subdiv2D_insert_vector(struct Subdiv2DPtr ptr, struct TensorWrapper ptvec);

extern "C"
struct Vec3iWrapper Subdiv2D_locate(struct Subdiv2DPtr ptr, struct Point2fWrapper pt);

extern "C"
struct Point2fPlusInt Subdiv2D_findNearest(struct Subdiv2DPtr ptr, struct Point2fWrapper pt);

extern "C"
struct TensorWrapper Subdiv2D_getEdgeList(struct Subdiv2DPtr ptr);

extern "C"
struct TensorWrapper Subdiv2D_getTriangleList(struct Subdiv2DPtr ptr);

extern "C"
struct TensorArray Subdiv2D_getVoronoiFacetList(struct Subdiv2DPtr ptr, struct TensorWrapper idx);

extern "C"
struct Point2fPlusInt Subdiv2D_getVertex(struct Subdiv2DPtr ptr, int vertex);

extern "C"
int Subdiv2D_getEdge(struct Subdiv2DPtr ptr, int edge, int nextEdgeType);

extern "C"
int Subdiv2D_nextEdge(struct Subdiv2DPtr ptr, int edge);

extern "C"
int Subdiv2D_rotateEdge(struct Subdiv2DPtr ptr, int edge, int rotate);

extern "C"
int Subdiv2D_symEdge(struct Subdiv2DPtr ptr, int edge);

extern "C"
struct Point2fPlusInt Subdiv2D_edgeOrg(struct Subdiv2DPtr ptr, int edge);

extern "C"
struct Point2fPlusInt Subdiv2D_edgeDst(struct Subdiv2DPtr ptr, int edge);

extern "C"
struct LineIteratorPtr LineIterator_ctor(
        struct TensorWrapper img, struct PointWrapper pt1, struct PointWrapper pt2,
        int connectivity, bool leftToRight);

extern "C"
void LineIterator_dtor(struct LineIteratorPtr ptr);

extern "C"
int LineIterator_count(struct LineIteratorPtr ptr);

extern "C"
struct PointWrapper LineIterator_pos(struct LineIteratorPtr ptr);

extern "C"
void LineIterator_incr(struct LineIteratorPtr ptr);

extern "C"
struct TensorWrapper addWeighted(
        struct TensorWrapper src1, double alpha, struct TensorWrapper src2, double beta,
        double gamma, struct TensorWrapper dst, int dtype);