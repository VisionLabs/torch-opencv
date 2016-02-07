local cv = require 'cv._env'
local ffi = require 'ffi'

ffi.cdef[[
struct calibrateCameraRetval {
    double retval;
    struct TensorArray rvecs, tvecs, intrinsics;
};

struct decomposeHomographyMatRetval {
   int val;
   struct TensorArray rotations, translations, normals;
};

struct TensorArrayPlusRectArrayPlusFloat {
    struct TensorArray tensors;
    struct RectArray rects;
    float val;
};

struct calibrateCameraRetval calibrateCamera(
	struct TensorArray objectPoints, struct TensorArray imagePoints,
	struct SizeWrapper imageSize, struct TensorWrapper cameraMatrix,
	struct TensorWrapper distCoeffs, struct TensorArray rvecs,
	struct TensorArray tvecs, int flags, struct TermCriteriaWrapper criteria);

struct TensorWrapper calibrationMatrixValues(
	struct TensorWrapper cameraMatrix,
	struct SizeWrapper imageSize,
	double apertureWidth, double apertureHeight);

struct TensorArray composeRT(
	struct TensorWrapper rvec1, struct TensorWrapper tvec1, struct TensorWrapper rvec2,
	struct TensorWrapper tvec2, struct TensorWrapper rvec3, struct TensorWrapper tvec3,
	struct TensorWrapper dr3dr1, struct TensorWrapper dr3dt1, struct TensorWrapper dr3dr2,
	struct TensorWrapper dr3dt2, struct TensorWrapper dt3dr1, struct TensorWrapper dt3dt1,
	struct TensorWrapper dt3dr2, struct TensorWrapper dt3dt2);

struct TensorWrapper computeCorrespondEpilines(
	struct TensorWrapper points, int whichImage, struct TensorWrapper F,
	struct TensorWrapper lines);

struct TensorWrapper convertPointsFromHomogeneous(
	struct TensorWrapper src, struct TensorWrapper dst);

struct TensorWrapper convertPointsHomogeneous(
	struct TensorWrapper src, struct TensorWrapper dst);

struct TensorWrapper convertPointsToHomogeneous(
	struct TensorWrapper src, struct TensorWrapper dst);

struct TensorArray correctMatches(
	struct TensorWrapper F, struct TensorWrapper points1,
	struct TensorWrapper points2, struct TensorWrapper newPoints1,
	struct TensorWrapper newPoints2);

struct TensorArray decomposeEssentialMat(
	struct TensorWrapper E, struct TensorWrapper R1,
	struct TensorWrapper R2, struct TensorWrapper t);

struct decomposeHomographyMatRetval decomposeHomographyMat(
	struct TensorWrapper H, struct TensorWrapper K,
	struct TensorArray rotations, struct TensorArray translations,
	struct TensorArray normals);

struct TensorArray decomposeProjectionMatrix(
	struct TensorWrapper projMatrix, struct TensorWrapper cameraMatrix,
	struct TensorWrapper rotMatrix, struct TensorWrapper transVect,
	struct TensorWrapper rotMatrixX, struct TensorWrapper rotMatrixY,
	struct TensorWrapper rotMatrixZ, struct TensorWrapper eulerAngles);

void drawChessboardCorners(
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper corners, bool patternWasFound);

struct TensorArrayPlusInt estimateAffine3D(
	struct TensorWrapper src, struct TensorWrapper dst,
	struct TensorWrapper out, struct TensorWrapper inliers,
	double ransacThreshold, double confidence);

void filterSpeckles(
	struct TensorWrapper img, double newVal, int maxSpeckleSize,
	double maxDiff, struct TensorWrapper buf);
 
void find4QuadCornerSubpix(
	struct TensorWrapper img, struct TensorWrapper corners,
	struct SizeWrapper region_size);

struct TensorWrapper findChessboardCorners(
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper corners, int flags);

struct TensorPlusBool findCirclesGrid(
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper centers, int flags,
	struct SimpleBlobDetectorPtr blobDetector);

struct TensorWrapper findEssentialMat(
	struct TensorWrapper points1, struct TensorWrapper points2,
	double focal, struct Point2dWrapper pp, int method, double prob,
	double threshold, struct TensorWrapper mask);

struct TensorWrapper findFundamentalMat(
	struct TensorWrapper points1, struct TensorWrapper points2, int method,
	double param1, double param2, struct TensorWrapper mask);

struct TensorWrapper findFundamentalMat2(
	struct TensorWrapper points1, struct TensorWrapper points2,
	struct TensorWrapper mask, int method, double param1, double param2);

struct TensorWrapper findHomography(
	struct TensorWrapper srcPoints, struct TensorWrapper dstPoints,
	int method, double ransacReprojThreshold, struct TensorWrapper mask,
	const int maxIters, const double confidence);

struct TensorWrapper findHomography2(
	struct TensorWrapper srcPoints, struct TensorWrapper dstPoints,
	struct TensorWrapper mask, int method, double ransacReprojThreshold);

struct TensorPlusRect getOptimalNewCameraMatrix(
	struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
	struct SizeWrapper imageSize, double alpha, struct SizeWrapper newImgSize,
	bool centerPrincipalPoint);

struct RectWrapper getValidDisparityROI(
	struct RectWrapper roi1, struct RectWrapper roi2,
	int minDisparity, int numberOfDisparities, int SADWindowSize);

struct TensorWrapper initCameraMatrix2D(
	struct TensorArray objectPoints, struct TensorArray imagePoints,
   	struct SizeWrapper imageSize, double aspectRatio);

struct TensorArray matMulDeriv(
	struct TensorWrapper A, struct TensorWrapper B,
	struct TensorWrapper dABdA, struct TensorWrapper dABdB);

struct TensorArray projectPoints(
	struct TensorWrapper objectPoints, struct TensorWrapper rvec,
	struct TensorWrapper tvec, struct TensorWrapper cameraMatrix,
	struct TensorWrapper distCoeffs, struct TensorWrapper imagePoints,
	struct TensorWrapper jacobian, double aspectRatio);

struct TensorArrayPlusInt recoverPose(
	struct TensorWrapper E, struct TensorWrapper points1,
	struct TensorWrapper points2, struct TensorWrapper R,
	struct TensorWrapper t, double focal,
	struct Point2dWrapper pp, struct TensorWrapper mask);

struct TensorArrayPlusRectArrayPlusFloat rectify3Collinear(
	struct TensorWrapper cameraMatrix1, struct TensorWrapper distCoeffs1,
	struct TensorWrapper cameraMatrix2, struct TensorWrapper distCoeffs2,
	struct TensorWrapper cameraMatrix3, struct TensorWrapper distCoeffs3,
	struct TensorArray imgpt1, struct TensorArray imgpt3,
	struct SizeWrapper imageSize, struct TensorWrapper R12,
	struct TensorWrapper T12, struct TensorWrapper R13,
	struct TensorWrapper T13, double alpha,
	struct SizeWrapper newImgSize, int flags);

struct TensorWrapper reprojectImageTo3D(
	struct TensorWrapper disparity, struct TensorWrapper _3dImage,
	struct TensorWrapper Q, bool handleMissingValues, int ddepth);

struct TensorArray Rodrigues(
	struct TensorWrapper src, struct TensorWrapper dst, struct TensorWrapper jacobian);

struct TensorArrayPlusVec3d RQDecomp3x3(
	struct TensorWrapper src, struct TensorWrapper mtxR, struct TensorWrapper mtxQ,
	struct TensorWrapper Qx, struct TensorWrapper Qy, struct TensorWrapper Qz);

struct TensorArrayPlusBool solvePnP(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints,
	struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
	struct TensorWrapper rvec, struct TensorWrapper tvec,
	bool useExtrinsicGuess, int flags);

struct TensorArrayPlusBool solvePnPRansac(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints,
	struct TensorWrapper cameraMatrix, struct TensorWrapper distCoeffs,
	struct TensorWrapper rvec, struct TensorWrapper tvec,
	bool useExtrinsicGuess, int iterationsCount, float reprojectionError,
	double confidence, struct TensorWrapper inliers, int flags);

struct TensorArrayPlusDouble stereoCalibrate(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints1,
	struct TensorWrapper imagePoints2, struct TensorWrapper cameraMatrix1,
	struct TensorWrapper distCoeffs1, struct TensorWrapper cameraMatrix2,
	struct TensorWrapper distCoeffs2, struct SizeWrapper imageSize,
	struct TensorWrapper R, struct TensorWrapper T,
	struct TensorWrapper E, struct TensorWrapper F,
	int flags, struct TermCriteriaWrapper criteria);

struct RectArray stereoRectify(
	struct TensorWrapper cameraMatrix1, struct TensorWrapper distCoeffs1,
	struct TensorWrapper cameraMatrix2, struct TensorWrapper distCoeffs2,
	struct SizeWrapper imageSize, struct TensorWrapper R,
	struct TensorWrapper T, struct TensorWrapper R1,
	struct TensorWrapper R2, struct TensorWrapper P1,
	struct TensorWrapper P2, struct TensorWrapper Q,
	int flags, double alpha, struct SizeWrapper newImageSize);

struct TensorArrayPlusBool stereoRectifyUncalibrated(
	struct TensorWrapper points1, struct TensorWrapper points2,
	struct TensorWrapper F, struct SizeWrapper imgSize,
	struct TensorWrapper H1, struct TensorWrapper H2, double threshold);

struct TensorWrapper triangulatePoints(
	struct TensorWrapper projMatr1, struct TensorWrapper projMatr2,
	struct TensorWrapper projPoints1, struct TensorWrapper projPoints2);

struct TensorWrapper validateDisparity(
	struct TensorWrapper disparity, struct TensorWrapper cost,
        int minDisparity, int numberOfDisparities, int disp12MaxDisp);

struct calibrateCameraRetval fisheye_calibrate(
	struct TensorArray objectPoints, struct TensorArray imagePoints,
	struct SizeWrapper imageSize, struct TensorWrapper K,
	struct TensorWrapper D, struct TensorArray rvecs,
	struct TensorArray tvecs, int flags, struct TermCriteriaWrapper criteria);

struct TensorWrapper fisheye_distortPoints(
	struct TensorWrapper undistorted, struct TensorWrapper distorted,
	struct TensorWrapper K, struct TensorWrapper D, double alpha);

struct TensorWrapper fisheye_estimateNewCameraMatrixForUndistortRectify(
	struct TensorWrapper K, struct TensorWrapper D,
	struct SizeWrapper image_size, struct TensorWrapper R,
	struct TensorWrapper P, double balance,
	struct SizeWrapper new_size, double fov_scale);

struct TensorArray fisheye_initUndistortRectifyMap(
	struct TensorWrapper K, struct TensorWrapper D,
	struct TensorWrapper R, struct TensorWrapper P,
	struct SizeWrapper size, int m1type,
	struct TensorWrapper map1, struct TensorWrapper map2);

struct TensorArray fisheye_projectPoints2(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints,
	struct TensorWrapper rvec, struct TensorWrapper tvec,
	struct TensorWrapper K, struct TensorWrapper D, double alpha,
	struct TensorWrapper jacobian);

struct TensorArrayPlusDouble fisheye_stereoCalibrate(
	struct TensorWrapper objectPoints, struct TensorWrapper imagePoints1,
	struct TensorWrapper imagePoints2, struct TensorWrapper K1,
	struct TensorWrapper D1, struct TensorWrapper K2,
	struct TensorWrapper D2, struct SizeWrapper imageSize,
	struct TensorWrapper R, struct TensorWrapper T,
	int flags, struct TermCriteriaWrapper criteria);

struct TensorArray fisheye_stereoRectify(
	struct TensorWrapper K1, struct TensorWrapper D1,
	struct TensorWrapper K2, struct TensorWrapper D2,
	struct SizeWrapper imageSize, struct TensorWrapper R,
	struct TensorWrapper tvec, struct TensorWrapper R1,
	struct TensorWrapper R2, struct TensorWrapper P1,
	struct TensorWrapper P2, struct TensorWrapper Q,
	int flags, struct SizeWrapper newImageSize,
	double balance, double fov_scale);

struct TensorWrapper fisheye_undistortImage(
	struct TensorWrapper distorted, struct TensorWrapper undistorted,
	struct TensorWrapper K, struct TensorWrapper D,
	struct TensorWrapper Knew, struct SizeWrapper new_size);

struct TensorWrapper fisheye_undistortPoints(
	struct TensorWrapper distorted, struct TensorWrapper undistorted,
	struct TensorWrapper K, struct TensorWrapper D,
	struct TensorWrapper R, struct TensorWrapper P);

void test(
	struct TensorArray imgs);
]]

local C = ffi.load(cv.libPath('calib3d'))

----------------------
function cv.test(t)
    local argRules = {
        {"imgs", required =  true}}
    local imgs = cv.argcheck(t, argRules)
    cv.wrap_tensors(imgs)
end


----------------------

function cv.calibrateCamera(t)
    local argRules = {
        {"objectPoints", required = true},
        {"imagePoints", required = true},
        {"imageSize", required = true, operator = cv.Size},
        {"cameraMatrix", default = nil}, 
        {"distCoeffs", default = nil},
        {"rvecs", default = nil},
        {"tvecs", default = nil},
        {"flag", default = 0},
        {"criteria", default =
			{cv.TERM_CRITERIA_COUNT+cv.TERM_CRITERIA_EPS, 30, cv.DBL_EPSILON},
			operator = cv.TermCriteria}
	}
    local objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs,
      		rvecs, tvecs, flag, criteria = cv.argcheck(t, argRules)
    local result = C.calibrateCamera(
			cv.wrap_tensors(objectPoints), cv.wrap_tensors(imagePoints),
			imageSize, cv.wrap_tensor(cameraMatrix), cv.wrap_tensor(distCoeffs),
			cv.wrap_tensors(rvecs), cv.wrap_tensors(tvecs), flag, criteria)
    return result.retval, cv.unwrap_tensors(result.intrinsics),
           cv.unwrap_tensors(result.rvecs, true),
           cv.unwrap_tensors(result.tvecs, true) 
end 

function cv.calibrationMatrixValues(t)
    local argRules = {
        {"cameraMatrix", required = true},
        {"imageSize", required = true},
        {"apertureWidth", required = true},
        {"apertureHeight", required = true}}
    local cameraMatrix, imageSize, apertureWidth, apertureHeight = cv.argcheck(t, argRules)
    local result = C.calibrationMatrixValues(
			cv.wrap_tensor(cameraMatrix), imageSize, apertureWidth, apertureHeight)
    local retval = cv.unwrap_tensors(result)
    return retval[1][1], retval[2][1], retval[3][1],
           cv.Point2d(retval[4][1], retval[5][1]), retval[6][1]
end

function cv.composeRT(t)
    local argRules = {
        {"rvec1", required = true},
        {"tvec1", required = true},
        {"rvec2", required = true},
        {"tvec2", required = true},
        {"rvec3", default = nil},
        {"tvec3", default = nil},
        {"dr3dr1", default = nil},
        {"dr3dt1", default = nil},
        {"dr3dr2", default = nil},
        {"dr3dt2", default = nil},
        {"dt3dr1", default = nil},
        {"dt3dt1", default = nil},
        {"dt3dr2", default = nil},
        {"dt3dt2", default = nil}}
    local rvec1, tvec1, rvec2, tvec2, rvec3, tvec3, dr3dr1, dr3dt1, dr3dr2,
		dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2 = cv.argcheck(t, argRules)
    local result = C.composeRT(
		rvec1, tvec1, rvec2, tvec2, rvec3, tvec3, dr3dr1, dr3dt1,
		dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2)
    return cv.unwrap_tensors(result)
end   

function cv.computeCorrespondEpilines(t)
    local argRules = {
        {"points", required = true},
        {"whichImage", required = true},
        {"F", required = true},
        {"lines", default = nil}}
    local points, whichImage, F, lines = cv.argcheck(t, argRules)
    local result = C.computeCorrespondEpilines(
				cv.wrap_tensor(points), whichImage,
				cv.wrap_tensor(F), cv.wrap_tensor(lines))
    return cv.unwrap_tensors(result)
end

function cv.convertPointsFromHomogeneous(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil}}
    local src, dst = cv.argcheck(t, argRules)
    local result = C.convertPointsFromHomogeneous(cv.wrap_tensor(src), cv.wrap_tensor(dst))
    return cv.unwrap_tensors(result)
end

function cv.convertPointsHomogeneous(t)
    local argRules = {
        {"src", required = true},
        {"dst", required = true}}
    local src, dst = cv.argcheck(t, argRules)
    local result = C.convertPointsHomogeneous(cv.wrap_tensor(src), cv.wrap_tensor(dst))
    return cv.unwrap_tensors()
end

function cv.convertPointsToHomogeneous(t)
    local argRules = {
        {"src", required = true},
        {"dst", default = nil}}
    local src , dst = cv.argcheck(t, argRules)
    local result = C.convertPointsToHomogeneous(cv.wrap_tensor(src), cv.wrap_tensor(dst))
    return cv.unwrap_tensors(result)
end

function cv.correctMatches(t)
    local argRules = {
        {"F", required = true},
        {"points1", required = true},
        {"points2", required = true},
        {"newPoints1", default = nil},
        {"newPoints2", default = nil}}
    local F, points1, points2, newPoints1, newPoints2 = cv.argcheck(t, argRules)
    local result = C.correctMatches(
			cv.wrap_tensor(F), cv.wrap_tensor(points1), cv.wrap_tensor(points2),
			cv.wrap_tensor(newPoints1), cv.wrap_tensor(newPoints2))
    return unwrap_tensors(result)
end

function cv.decomposeEssentialMat(t)
    local argRules = {
        {"E", required = true},
        {"R1", default = nil},
        {"R2", default = nil},
        {"t", default = nil}}
    local E, R1, R2, t = cv.argcheck(t, argRules)
    local result = C.decomposeEssentialMat(cv.wrap_tensor(E), cv.wrap_tensor(R1),
					   cv.wrap_tensor(R2), cv.wrap_tensor(t))
    return unwrap_tensors(result)
end

function cv.decomposeHomographyMat(t)
    local argRules = {
       {"H", required = true},
       {"K", required = true},
       {"rotations", default = nil},
       {"translations", default = nil},
       {"normals", default = nil}}
    local H, K, rotations, translations, normals = cv.argcheck(t, argRules)
    local result = C.decomposeHomographyMat(
				cv.wrap_tensor(H), cv.wrap_tensor(K),
				cv.wrap_tensors(rotations), cv.wrap_tensors(translations),
				cv.wrap_tensors(normals))
    return result.val, unwrap_tensors(result.rotations, true),
           unwrap_tensors(result.translations, true), unwrap_tensors(result.normals, true)
end

function cv.decomposeProjectionMatrix(t)
    local argRules = {
        {"projMatrix", required = true},
        {"cameraMatrix", default = nil},
        {"rotMatrix", default = nil},
        {"transVect", default = nil},
        {"rotMatrixX", default = nil},
        {"rotMatrixY", default = nil},
        {"rotMatrixZ", default = nil},
        {"eulerAngles", default = nil}}
    local projMatrix, cameraMatrix, rotMatrix, transVect, rotMatrixX,
	  rotMatrixY, rotMatrixZ, eulerAngles = cv.argcheck(t, argRules)
    local result = C.decomposeProjectionMatrix(
			cv.wrap_tensor(projMatrix), cv.wrap_tensor(cameraMatrix),
			cv.wrap_tensor(rotMatrix), cv.wrap_tensor(transVect),
			cv.wrap_tensor(rotMatrixX), cv.wrap_tensor(rotMatrixY),
			cv.wrap_tensor(rotMatrixZ), cv.wrap_tensor(eulerAngles))
    return unwrap_tensors(result)
end 

function cv.drawChessboardCorners(t)
    local argRules = {
        {"image", required = true},
        {"patternSize", required = true, operator = cv.Size},
        {"corners", required = true},
        {"patternWasFound", default = true}}
    local image, patternSize, corners, patternWasFound = cv.argcheck(t, argRules)
    C.drawChessboardCorners(cv.wrap_tensor(image), patternSize, cv.wrap_tensor(corners), patternWasFound)
end

function cv.estimateAffine3D(t)
    local argRules = {
        {"src", required = true},
        {"dst", required = true},
        {"out", default = nil},
        {"inliers", default = nil},
        {"ransacThreshold", default = 3},
        {"confidence", default = 0.99}}
    local src, dst, out, inliers, ransacThreshold, confidence = cv.argcheck(t, argRules)
    local result = C.estimateAffine3D(
				cv.wrap_tensor(src), cv.wrap_tensor(dst), cv.wrap_tensor(out),
				cv.wrap_tensor(inliers), ransacThreshold, confidence)
    return unwrap_tensors(result)
end 

function cv.filterSpeckles(t)
    local argRules = {
        {"img", required = true},
        {"newVal", required = true},
        {"maxSpeckleSize", required = true},
        {"maxDiff", required = true},
        {"buf", default = nil}}
    local img, newVal, maxSpeckleSize, maxDiff, buf = cv.argcheck(t, argRules)
    C.filterSpeckles(
 		cv.wrap_tensor(img), newVal, maxSpeckleSize,
		maxDiff, cv.wrap_tensor(buf))
end

function cv.find4QuadCornerSubpix(t)
    local argRules = {
        {"img", required = true},
        {"corners", required = true},
        {"region_size", required = true, operator = cv.Size}}
    local img, corners, region_size = cv.argcheck(t, argRules)
    C.find4QuadCornerSubpix(
		cv.wrap_tensor(img), cv.wrap_tensor(corners), region_size)
end

function cv.findChessboardCorners(t)
    local argRules = {
        {"image", required = true},
        {"patternSize", required = true, operator = cv.Size},
        {"corners", default = nil},
        {"flags", default = cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_NORMALIZE_IMAGE}}
    local image, patternSize, corners, flags = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.findChessboardCorners(
			cv.wrap_tensor(image), patternSize,
			cv.wrap_tensor(corners), flags))
end

function cv.findCirclesGrid(t)
    local argRules = {
        {"image", required = true},
        {"patternSize", required = true, operator = cv.Size},
        {"centers", default = nil},
        {"flags", default = cv.CALIB_CB_SYMMETRIC_GRID},
        {"blobDetector", default = SimpleBlobDetector()}}
    local image, patternSize, centers, flags, blobDetector = cv.argcheck(t, argRules)
    local result = C.findCirclesGrid(
			cv.wrap_tensor(image), patternSize, cv.wrap_tensor(centers),
			flags, blobDetector)
    return result.val, cv.unwrap_tensors(result.tensor)
end 

function cv.findEssentialMat(t)
    local argRules = {
        {"points1", required = true},
        {"points2", required = true},
        {"focal", default = 1.0},
        {"pp", operator = cv.Point2d, default = cv.Point2d(0,0)},
        {"method", default = cv.RANSAC},
        {"prob", default = 0.999},
        {"threshold", default = 1.0},
        {"mask", default = nil}}
    local points1, points2, focal, pp, method, prob, threshold, mask = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
			C.findEssentialMat(
				cv.wrap_tensor(points1), cv.wrap_tensor(points2), focal,
				pp, method, prob, threshold, cv.wrap_tensor(mask)))  
end    

function cv.findFundamentalMat(t)
    local argRules = {
        {"points1", required = true},
        {"points2", required = true},
        {"method", default = cv.FM_RANSAC},
        {"param1", default = 3.0},
        {"param2", default = 0.99},
        {"mask", default = nil}}
    local points1, points2, method, param1, param2, mask = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.findFundamentalMat(
			cv.wrap_tensor(points1), cv.wrap_tensor(points2), method,
			param1, param2, cv.wrap_tensor(mask)))
end

function cv.findFundamentalMat2(t)
    local argRules = {
	{"points1", required = true},
   	{"points2", required = true},
	{"mask", default = nil},
	{"method", default = cv.FM_RANSAC},
  	{"param1", default = 3.0},
	{"param2", default = 0.99}}
    local points1, points2, mask, method, param1, param2 = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.findFundamentalMat(
			cv.wrap_tensor(points1), cv.wrap_tensor(points2),
			cv.wrap_tensor(mask), method, param1, param2))
end

function cv.findHomography(t)
    local argRules = {
        {"srcPoints", required = true},
	{"dstPoints", required = true},
	{"method", default = 0},
	{"ransacReprojThreshold", default = 3},
	{"mask", default = nil},
	{"maxIters", default = 2000},
	{"confidence", default = 0.995}}
    local srcPoints, dstPoints, method, ransacReprojThreshold,
	  mask, maxIters, confidence = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(C.findHomography(
					cv.wrap_tensor(srcPoints), cv.wrap_tensor(dstPoints),
					method, ransacReprojThreshold, cv.wrap_tensor(mask),
					maxIters, confidence))
end

function cv.findHomography2(t)
    local argRules = {
        {"srcPoints", required = true},
        {"dstPoints", required = true},
        {"mask", required = true},
        {"method", default = 0},
        {"ransacReprojThreshold", default = 3}}
    local srcPoints, dstPoints, mask, method, ransacReprojThreshold = argcheck(t, argRules)
    return cv.unwrap_tensors(
			C.findHomography(
				cv.wrap_tensor(srcPoints), cv.wrap_tensor(dstPoints),
				cv.wrap_tensor(mask), method, ransacReprojThreshold))
end

function cv.getOptimalNewCameraMatrix(t)
    local argRules = {
       {"cameraMatrix", required = true},
       {"distCoeffs", requierd = true},
       {"imageSize", required = true, operator = cv.Size},
       {"alpha", required = true},
       {"newImgSize", default = cv.Size(), operator = cv.Size},
       {"centerPrincipalPoint", default = false}}
    local cameraMatrix, distCoeffs, imageSize, alpha,
          newImgSize, centerPrincipalPoint = cv.argcheck(t, argRules)
    local result  = C.getOptimalNewCameraMatrix(
				cv.wrap_tensor(cameraMatrix), cv.wrap_tensor(distCoeffs),
 				imageSize, alpha, newImgSize, centerPrincipalPoint)
    return cv.unwrap_tensors(result.tensor), result.rect
end

function cv.getValidDisparityROI(t)
    local argRules = {
        {"roi1", required = true},
        {"roi2", required = true},
        {"minDisparity", required = true},
        {"numberOfDisparities", required = true},
        {"SADWindowSize", required = true}}
    local roi1, roi2, minDisparity, numberOfDisparities, SADWindowSize = cv.argcheck(t, argRules)
    return C.getValidDisparityROI(roi1, roi2, minDisparity, numberOfDisparities, SADWindowSize)  
end

function cv.initCameraMatrix2D(t)
    local argRules = {
        {"objectPoints", required = true},
        {"imagePoints", required = true},
        {"imageSize", required = true, operator = cv.Size},
        {"aspectRatio", default = 1.0}}
    local objectPoints, imagePoints, imageSize, aspectRatio = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
			C.initCameraMatrix2D(
				cv.wrap_tensors(objectPoints), cv.wrap_tensors(imagePoints),
				imageSize, aspectRatio))
end

function matMulDeriv(t)
    local argRules = {
        {"A", required = true},
        {"B", required = true},
        {"dABdA", default = nil},
        {"dABdB", default = nil}}
    local A, B, dABdA, dABdB = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.matMulDeriv(
			cv.wrap_tensor(A), cv.wrap_tensor(B),
			cv.wrap_tensor(dABdA), cv.wrap_tensor(dABdB)))
end

function cv.projectPoints(t)
    local argRules = {
        {"objectPoints", required = true},
        {"rvec", required = true},
        {"tvec", required = true},
        {"cameraMatrix", required = true},
        {"distCoeffs", required = true},
        {"imagePoints", default = nil},
        {"jacobian", default = nil},
        {"aspectRatio", default = 0}}
    local objectPoints, rvec, tvec, cameraMatrix, distCoeffs,
          imagePoints, jacobian, aspectRatio = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.projectPoints(
			cv.wrap_tensor(objectPoints), cv.wrap_tensor(rvec),
			cv.wrap_tensor(tvec), cv.wrap_tensor(cameraMatrix),
			cv.wrap_tensor(distCoeffs), cv.wrap_tensor(imagePoints),
			cv.wrap_tensor(jacobian), aspectRatio))
end
        
function cv.recoverPose(t)
    local argRules = {
        {"E", required = true},
        {"points1", required = true},
        {"points2", required = true},
        {"R", default = nil},
        {"t", default = nil},
        {"focal", default = 1.0},
        {"Point2d", default = cv.Point2d(0,0), operator = cv.Point2d},
        {"mask", default = nil}}
    local E, points1, points2, R, t, focal, Point2d, mask = cv.argcheck(t, argRules)
    local result = C.recoverPose(
			cv.wrap_tensor(E), cv.wrap_tensor(points1),
			cv.wrap_tensor(points2), cv.wrap_tensor(R),
			cv.wrap_tensor(t), focal, Point2d, mask)
    return result.val, cv.unwrap_tensors(result.tensors)
end

function cv.rectify3Collinear(t)
    local argRules = {
        {"cameraMatrix1", required = true},
        {"distCoeffs1", required = true},
        {"cameraMatrix2", required = true},
        {"distCoeffs2", required= true},
        {"cameraMatrix3", required = true},
        {"distCoeffs3", required = true},
        {"imgpt1", required = true},
        {"imgpt3", required = true},
        {"imageSize", required = true, operator = cv.Size},
        {"R12", required = true},
        {"T12", required = true},
        {"R13", required = true},
        {"T13", required = true},
        {"alpha", required = true},
        {"newImgSize", required = true, operator = cv.Size},
        {"flags", required = true}}
    local cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, cameraMatrix3,
	  distCoeffs3, imgpt1, imgpt3, imageSize, R12, T12, R13, T13, alpha,
          newImgSize, flags = cv.argcheck(t, argRules)
    local result = C.rectify3Collinear(
			cv.wrap_tensor(cameraMatrix1), cv.wrap_tensor(distCoeffs1),
			cv.wrap_tensor(cameraMatrix2), cv.wrap_tensor(distCoeffs2),
			cv.wrap_tensor(cameraMatrix3), cv.wrap_tensor(distCoeffs3),
			cv.wrap_tensor(imgpt1), cv.wrap_tensors(imgpt3), imageSize,
			cv.wrap_tensor(R12), cv.wrap_tensor(T12), cv.wrap_tensor(R13),
			cv.wrap_tensor(T13), alpha, newImgSize, flags)
    return result.val, cv.unwrap_tensors(result.tensors), cv.gcarray(result.rects)
end

function cv.reprojectImageTo3D(t)
    local argRules = {
        {"disparity", required = true},
        {"_3dImage", default = nil},
        {"Q", required = true},
        {"handleMissingValues", default = false},
        {"ddepth", default = -1}}
    local disparity, _3dImage, Q, handleMissingValues, ddepth = cv.argcheck(t, argRules)
    return cv_unwrap_tensors(
			C.reprojectImageTo3D(
				cv.wrap_tensor(disparity), cv.wrap_tensor(_3dImage),
				cv.wrap_tensor(Q), handleMissingValues, ddepth))
end

function cv.Rodrigues(t)
    local argRules = {
        {"src", required = true},
        {"dst", required = true},
        {"jacobian", required = true}}
    local src, dst, jacobian = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.Rodrigues(
			cv.wrap_tensor(src), cv.wrap_tensor(dst),
			cv.wrap_tensor(jacobian)))
end

function cv.RQDecomp3x3(t)
    local argRules = {
        {"src", required = true},
        {"mtxR", default = nil},
        {"mtxQ", default = nil},
        {"Qx", default = nil},
        {"Qy", default = nil},
        {"Qz", default = nil}}
    local src, mtxR, mtxQ, Qx, Qy, Qz = cv.argcheck(t, argRules)
    local result = C.RQDecomp3x3(
			cv.wrap_tensor(src), cv.wrap_tensor(mtxR),
			cv.wrap_tensor(mtxQ), cv.wrap_tensor(Qx),
			cv.wrap_tensor(Qy), cv.wrap_tensor(Qz))
    return result.vec3d, cv.unwrap_tensors(result.tensors)
end

function cv.solvePnP(t)
    local argRules = {
        {"objectPoints", required = true},
        {"imagePoints", required = true},
        {"cameraMatrix", required = true},
        {"distCoeffs", required = true},
        {"rvec", default = nil},
        {"tvec", default = nil},
        {"useExtrinsicGuess", default = false},
        {"flags", default = cv.SOLVEPNP_ITERATIVE}}
    local objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec,
          tvec, useExtrinsicGuess, flags = cv.argcheck(t, argRules)
    local result = C.solvePnP(
			cv.wrap_tensor(objectPoints), cv.wrap_tensor(imagePoints),
             		cv.wrap_tensor(cameraMatrix), cv.wrap_tensor(distCoeffs),
               		cv.wrap_tensor(rvec), cv.wrap_tensor(tvec),
			useExtrinsicGuess, flags)
    return result.val, cv.unwrap_tensors(result.tensors)
end

function cv.solvePnPRansac(t)
    local argRules = {
        {"objectPoints", required = true},
        {"imagePoints", required = true},
        {"cameraMatrix", required = true},
        {"distCoeffs", required = true},
        {"rvec", default = nil},
        {"tvec", default = nil},
        {"useExtrinsicGuess", default = false},
        {"iterationsCount", default = 10},
        {"reprojectionError", default = 8.0},
        {"confidence", default = 0.99},
        {"inliers", default = nil},
        {"flags", default = cv.SOLVEPNP_ITERATIVE}}
    local objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec,
          tvec, useExtrinsicGuess, iterationsCount,reprojectionError,
          confidence, inliers, flags = cv.argcheck(t, argRules)
    local result = C.solvePnPRansac(
			cv.wrap_tensor(objectPoints), cv.wrap_tensor(imagePoints),
			cv.wrap_tensor(cameraMatrix), cv.wrap_tensor(distCoeffs),
			cv.wrap_tensor(rvec), cv.wrap_tensor(tvec), useExtrinsicGuess, 				iterationsCount, reprojectionError, confidence,
			cv.wrap_tensor(inliers), flags)
    return result.val, cv.unwrap_tensors(result.tensors)
end

function cv.stereoCalibrate(t)
    local argRules = {
        {"objectPoints", required = true},
        {"imagePoints1", required = true},
        {"imagePoints2", required = true},
        {"cameraMatrix1", default = nil},
        {"distCoeffs1", default = nil},
        {"cameraMatrix2", default = nil},
        {"distCoeffs2", default = nil},
        {"imageSize", required = true, operator = cv.Size},
        {"R", default = nil},
        {"T", default = nil},
        {"E", default = nil},
        {"F", default = nil},
        {"flags", defauly = cv.CALIB_FIX_INTRINSIC},
        {"criteria", default =
		cv.TermCriteria(cv.TERM_CRITERIA_COUNT+cv.TERM_CRITERIA_EPS, 30, 1e-6),
		operator = cv.TermCriteria}}
    local objectPoints, imagePoints1, imagePoints2, cameraMatrix1,
          distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T,
          E, F, flags, criteria = cv.argcheck(t, argRules)
    local result = C.stereoCalibrate(
			cv.wrap_tensor(objectPoints), cv.wrap_tensor(imagePoints1),
			cv.wrap_tensor(imagePoints2), cv.wrap_tensor(cameraMatrix1),
			cv.wrap_tensor(distCoeffs1), cv.wrap_tensor(cameraMatrix2),
			cv.wrap_tensor(distCoeffs2), imageSize, cv.wrap_tensor(R),
			cv.wrap_tensor(T), cv.wrap_tensor(E), cv.wrap_tensor(F),
			flags, criteria)
    return result.val, cv.unwrap_tensors(result.tensors)
end

function cv.stereoRectify(t)
    local argRules = {
        {"cameraMatrix1", required = true},
        {"distCoeffs1", required = true},
        {"cameraMatrix2", required = true},
        {"distCoeffs2", required = true},
        {"imageSize", required = true, operator = cv.Size},
        {"R", required = true},
        {"T", required = true},
        {"R1", required = true},
        {"R2", required = true},
        {"P1", required = true},
        {"P2", required = true},
        {"Q", required = true},
        {"flags", default = cv.CALIB_ZERO_DISPARITY},
        {"alpha", default = -1},
        {"newImageSize", default = cv.Size(), operator = cv.Size}}
    local cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
          imageSize, R, T, R1, R2, P1, P2, Q, flags, alpha,
          newImageSize = cv.argcheck(t, argRules)
    return cv.gcarray(
		C.stereoRectify(
			cv.wrap_tensor(cameraMatrix1), cv.wrap_tensor(distCoeffs1),
			cv.wrap_tensor(cameraMatrix2), cv.wrap_tensor(distCoeffs2),
			imageSize, cv.wrap_tensor(R), cv.wrap_tensor(T), cv.wrap_tensor(R1),
			cv.wrap_tensor(R2), cv.wrap_tensor(P1), cv.wrap_tensor(P2),
			cv.wrap_tensor(Q), flags, alpha, newImageSize))
end

function cv.stereoRectifyUncalibrated(t)
    local argRules = {
        {"points1", requred = true},
        {"points2", required = true},
        {"F", required = true},
        {"imgSize", required = true},
        {"H1", required = true},
        {"H2", required = true},
        {"threshold", default = 5}}
    local points1, points2, F, imgSize, H1, H2, threshold = cv.argcheck(t, argRules)
    local result = C.stereoRectifyUncalibrated(
				cv.wrap_tensor(points1), cv.wrap_tensor(points2),
				cv.wrap_tensor(F), imgSize, cv.wrap_tensor(H1),
				cv.wrap_tensor(H2), threshold);
    return result.val, cv.unwrap_tensors(result.tensors)
end

function cv.triangulatePoints(t)
    local argRules = {
        {"projMatr1", required = true},
        {"projMatr2", required = true},
        {"projPoints1", required = true},
        {"projPoints2", required = true}}
    local projMatr1, projMatr2, projPoints1, projPoints2 = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
			C.triangulatePoints(
				cv.wrap_tensor(projMatr1), cv.wrap_tensor(projMatr2),
				cv.wrap_tensor(projPoints1), cv.wrap_tensor(projPoints2))) 
end

function cv.validateDisparity(t)
    local argRules = {
        {"disparity", required = true},
        {"cost", required = true},
        {"minDisparity", required = true},
        {"numberOfDisparities", required = true},
        {"disp12MaxDisp", default = 1}}
    local disparity, cost, minDisparity, numberOfDisparities,
          disp12MaxDisp = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.validateDisparity(
			cv.wrap_tensor(disparity), cv.wrap_tensor(cost),
			minDisparity, numberOfDisparities, disp12MaxDisp))
end

--******************Fisheye camera model***************

fisheye = {}
cv.fisheye = fisheye;

cv.fisheye.CALIB_USE_INTRINSIC_GUESS = 1
cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC = 2
cv.fisheye.CALIB_CHECK_COND = 4
cv.fisheye.CALIB_FIX_SKEW = 8
cv.fisheye.CALIB_FIX_K1 = 16
cv.fisheye.CALIB_FIX_K2 = 32
cv.fisheye.CALIB_FIX_K3 = 64
cv.fisheye.CALIB_FIX_K4 = 128
cv.fisheye.CALIB_FIX_INTRINSIC = 256

function cv.fisheye.calibrate(t)
    local argRules = {
        {"objectPoints", required = true},
        {"imagePoints", required = true},
        {"imageSize", required = true, operator = cv.Size},
        {"K", default = nil}, 
        {"D", default = nil},
        {"rvecs", default = nil},
        {"tvecs", default = nil},
        {"flag", default = 0},
        {"criteria", default =
		cv.TermCriteria(cv.TERM_CRITERIA_COUNT+cv.TERM_CRITERIA_EPS, 100, cv.DBL_EPSILON),
                operator = cv.TermCriteria}}
    local objectPoints, imagePoints, imageSize, K, D,
      		rvecs, tvecs, flag, criteria = cv.argcheck(t, argRules)
    local result = C.fisheye_calibrate(
			cv.wrap_tensors(objectPoints), cv.wrap_tensors(imagePoints),
			imageSize, cv.wrap_tensor(K), cv.wrap_tensor(D),
			cv.wrap_tensors(rvecs), cv.wrap_tensors(tvecs), flag, criteria)
    return result.retval, cv.unwrap_tensors(result.intrinsics),
           cv.unwrap_tensors(result.rvecs, true),
           cv.unwrap_tensors(result.tvecs, true) 
end 

function cv.fisheye.distortPoints(t)
    local argRules = {
        {"undistorted", required = true},
        {"distorted", default = nil},
        {"K", required = true},
        {"D", required = true},
        {"alpha", default = 0}}
    local undistorted, distorted, K, D, alpha = cv.argcheck(t, argRules)
    return unwrap_tensors(
		C.fisheye_distortPoints(
			cv.wrap_tensor(undistorted), cv.wrap_tensor(distorted),
			cv.wrap_tensor(K), cv.wrap_tensor(D), alpha))
end

function cv.fisheye.estimateNewCameraMatrixForUndistortRectify(t)
    local argRules = {
        {"K", required = true},
        {"D", required = true},
        {"image_size", required = true, operator = cv.Size},
        {"R", required = true},
        {"P", default = nil},
        {"balance", default = 0.0},
        {"new_size", default = cv.Size(), operator = cv.Size},
        {"fov_scale", default = 1.0}}
    local K, D, image_size, R, P, balance, new_size,
          fov_scale = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.fisheye_estimateNewCameraMatrixForUndistortRectify(
			cv.wrap_tensor(K), cv.wrap_tensor(D), image_size,
			cv.wrap_tensor(R), cv.wrap_tensor(P), balance,
			new_size, fov_scale))
end

function cv.fisheye.initUndistortRectifyMap(t)
    local argRules = {
        {"K", required = true},
        {"D", required = true},
        {"R", required = true},
        {"P", required = true},
        {"size", required = true, operator = cv.Size},
        {"m1type", required = true},
        {"map1", default = nil},
        {"map2", default = nil}}
    local K, D, R, P, size, m1type, map1, map2 = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.fisheye_initUndistortRectifyMap(
			cv.wrap_tensor(K), cv.wrap_tensor(D), cv.wrap_tensor(R),
			cv.wrap_tensor(P), size, m1type, cv.wrap_tensor(map1),
			cv.wrap_tensor(map2)));
end

function cv.fisheye.projectPoints2(t)
    local argRules = {
        {"objectPoints", required = true},
        {"imagePoints", default = nil},
        {"rvec", required = true},
        {"tvec", required = true},
        {"K", required = true},
        {"D", required = true},
        {"alpha", default = 0},
        {"jacobian", default = nil}}
    local objectPoints, imagePoints, rvec, tvec, K,
          D, alpha, jacobian = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.fisheye_projectPoints2(
			cv.wrap_tensor(objectPoints), cv.wrap_tensor(imagePoints),
			cv.wrap_tensor(rvec), cv.wrap_tensor(tvec), cv.wrap_tensor(K),
			cv.wrap_tensor(D), alpha, jacobian))
end

function cv.fisheye.stereoCalibrate(t)
    local argRules = {
        {"objectPoints", required = true},
        {"imagePoints1", required = true},
        {"imagePoints2", required = true},
        {"K1", default = nil},
        {"D1", default = nil},
        {"K2", default = nil},
        {"D2", default = nil},
        {"imageSize", required = true, operator = cv.Size},
        {"R", default = nil},
        {"T", default = nil},
        {"flags", defauly = cv.fisheye.CALIB_FIX_INTRINSIC},
        {"criteria", default =
		cv.TermCriteria(cv.TERM_CRITERIA_COUNT+cv.TERM_CRITERIA_EPS, 100, cv.DBL_EPSILON),
		operator = cv.TermCriteria}}
    local objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2,
          imageSize, R, T, flags, criteria = cv.argcheck(t, argRules)
    local result = C.fisheye_stereoCalibrate(
			cv.wrap_tensor(objectPoints), cv.wrap_tensor(imagePoints1),
			cv.wrap_tensor(imagePoints2), cv.wrap_tensor(K1),
			cv.wrap_tensor(D1), cv.wrap_tensor(K2),
			cv.wrap_tensor(D2), imageSize, cv.wrap_tensor(R),
			cv.wrap_tensor(T), flags, criteria)
    return result.val, cv.unwrap_tensors(result.tensors)
end

function cv.fisheye.stereoRectify(t)
    local argRules = {
        {"K1", required = true},
        {"D1", required = true},
        {"K2", required = true},
        {"D2", required = true},
        {"imageSize", required = true, operator = cv.Size},
        {"R", required = true},
        {"tvec", required = true},
        {"R1", default = nil},
        {"R2", default = nil},
        {"P1", default = nil},
        {"P2", default = nil},
        {"Q", default = nil},
        {"flags", required = true},
        {"newImageSize", default = cv.Size(), operator = cv.Size},
        {"balance", default = 0.0},
        {"fov_scale", default = 1.0}}
    local K1, D1, K2, D, imageSize, R, tvec, R1, R2, P1, P2, Q, flags,
          newImageSize, balance, fov_scale = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.fisheye_stereoRectify(
			cv.wrap_tensor(K1), cv.wrap_tensor(D1),
			cv.wrap_tensor(K2), cv.wrap_tensor(D2),
			imageSize, cv.wrap_tensor(R), cv.wrap_tensor(tvec), cv.wrap_tensor(R1),
			cv.wrap_tensor(R2), cv.wrap_tensor(P1), cv.wrap_tensor(P2),
			cv.wrap_tensor(Q), flags, newImageSize, balance, fov_scale))
end

function cv.fisheye.undistortImage(t)
    local argRules = {
        {"distorted", required = true},
        {"undistorted", default = nil},
        {"K", required = true},
        {"D", required = true},
        {"Knew", default = nil},
        {"new_size", default = cv.Size(), operator = cv.Size}}
    local distorted, undistorted, K, D, Knew, new_size = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.fisheye._ndistortImage(
			cv.wrap_tensor(distorted), cv.wrap_tensor(undistorted),
			cv.wrap_tensor(K), cv.wrap_tensor(D),
			cv.wrap_tensor(Knew), new_size))
end

function cv.fisheye.undistortPoints(t)
    local argRules = {
        {"distorted", required = true},
        {"undistorted", default = nil},
        {"K", required = true},
        {"D", required = true},
        {"R", default = nil},
        {"P", default = nil}}
    local distorted, undistorted, K, D, R, P = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.fisheye_undistortPoints(
			cv.wrap_tensor(distorted), cv.wrap_tensor(undistorted),
			cv.wrap_tensor(K), cv.wrap_tensor(D), cv.wrap_tensor(R),
			cv.wrap_tensor(P)))
end

--- ***************** Classes *****************

require 'cv.Classes'

local Classes = ffi.load(cv.libPath('Classes'))

ffi.cdef[[

struct TensorWrapper StereoMatcher_compute(
	struct StereoMatcherPtr ptr, struct TensorWrapper left,
	struct TensorWrapper right, struct TensorWrapper disparity);

int StereoMatcher_getBlockSize(
	struct StereoMatcherPtr ptr);

int StereoMatcher_getDisp12MaxDiff(
	struct StereoMatcherPtr ptr);

int StereoMatcher_getMinDisparity(
	struct StereoMatcherPtr ptr);

int StereoMatcher_getNumDisparities(
	struct StereoMatcherPtr ptr);

int StereoMatcher_getSpeckleRange(
	struct StereoMatcherPtr ptr);

int StereoMatcher_getSpeckleWindowSize(
	struct StereoMatcherPtr ptr);

void StereoMatcher_setBlockSize(
	struct StereoMatcherPtr ptr, int blockSize);

void StereoMatcher_setDisp12MaxDiff(
	struct StereoMatcherPtr ptr, int disp12MaxDiff);

void StereoMatcher_setMinDisparity(
	struct StereoMatcherPtr ptr, int minDisparity);

void StereoMatcher_setNumDisparities(
	struct StereoMatcherPtr ptr, int numDisparities);

void StereoMatcher_setSpeckleRange(
	struct StereoMatcherPtr ptr, int speckleRange);

void StereoMatcher_setSpeckleWindowSize(
	struct StereoMatcherPtr ptr, int speckleWindowSize);

struct StereoBMPtr StereoBM_ctor(
	int numDisparities, int blockSize);

int StereoBM_getPreFilterCap(
	struct StereoBMPtr ptr);

int StereoBM_getPreFilterSize(
	struct StereoBMPtr ptr);

int StereoBM_getPreFilterType(
	struct StereoBMPtr ptr);

struct RectWrapper StereoBM_getROI1(
	struct StereoBMPtr ptr);

struct RectWrapper StereoBM_getROI2(
	struct StereoBMPtr ptr);

int StereoBM_getSmallerBlockSize(
	struct StereoBMPtr ptr);

int StereoBM_getTextureThreshold(
	struct StereoBMPtr ptr);

int StereoBM_getUniquenessRatio(
	struct StereoBMPtr ptr);

void StereoBM_setPreFilterCap(
	struct StereoBMPtr ptr, int preFilterCap);

void StereoBM_setPreFilterSize(
	struct StereoBMPtr ptr, int preFilterSize);

void StereoBM_setPreFilterType(
	struct StereoBMPtr ptr, int preFilterType);

void StereoBM_setROI1(
	struct StereoBMPtr ptr, struct RectWrapper roi1);

void StereoBM_setROI2(
	struct StereoBMPtr ptr, struct RectWrapper roi2);

void StereoBM_setSmallerBlockSize(
	struct StereoBMPtr ptr, int blockSize);

void StereoBM_setTextureThreshold(
	struct StereoBMPtr ptr, int textureThreshold);

void StereoBM_setUniquenessRatio(
	struct StereoBMPtr ptr, int uniquenessRatio);

struct StereoSGBMPtr StereoSGBM_ctor(
	int minDisparity, int numDisparities, int blockSize,
	int P1, int P2, int disp12MaxDiff, int preFilterCap,
	int uniquenessRatio, int speckleWindowSize,
	int speckleRange, int mode);

int StereoSGBM_getMode(
	struct StereoSGBMPtr ptr);

int StereoSGBM_getP1(
	struct StereoSGBMPtr ptr);

int StereoSGBM_getP2(
	struct StereoSGBMPtr ptr);

int StereoSGBM_getPreFilterCap(
	struct StereoSGBMPtr ptr);

int StereoSGBM_getUniquenessRatio(
	struct StereoSGBMPtr ptr);

void StereoSGBM_setMode(
	struct StereoSGBMPtr ptr, int mode);

void StereoSGBM_setP1(
	struct StereoSGBMPtr ptr, int P1);

void StereoSGBM_setP2(
	struct StereoSGBMPtr ptr, int P2);

void StereoSGBM_setPreFilterCap(
	struct StereoSGBMPtr ptr, int preFilterCap);

void StereoSGBM_setUniquenessRatio(
	struct StereoSGBMPtr ptr, int uniquenessRatio);
]]

--StereoMatcher

do
    local StereoMatcher = torch.class('cv.StereoMatcher', 'cv.Algorithm', cv)

    function StereoMatcher:compute(t)
        local argRules = {
            {"left", required = true},
            {"right", required = true},
            {"disparity", default = nil}}
    local left, right, right = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
			C.StereoMatcher_compute(
				self.ptr, cv.wrap_tensor(left), cv.wrap_tensor(right),
				cv.wrap_tensor(disparity)))
    end

    function StereoMatcher:getBlockSize()
        return C.StereoMatcher_getBlockSize(self.ptr)
    end

    function StereoMatcher:getDisp12MaxDiff()
        return C.StereoMatcher_getDisp12MaxDiff(self.ptr)
    end

    function StereoMatcher:getMinDisparity()
        return C.StereoMatcher_getMinDisparity(self.ptr)
    end

    function StereoMatcher:getNumDisparities()
        return C.StereoMatcher_getNumDisparities(self.ptr)
    end

    function StereoMatcher:getSpeckleRange()
        return C.StereoMatcher_getSpeckleRange(self.ptr)
    end

    function StereoMatcher:getSpeckleWindowSize()
        return C.StereoMatcher_getSpeckleWindowSize(self.ptr)
    end

    function StereoMatcher:setBlockSize(t)
        local argRules = {
            {"setBlockSize", required = true}}
        local setBlockSize = cv.argcheck(t, argRules)
        C.StereoMatcher_setBlockSize(self.ptr, setBlockSize)
    end

    function StereoMatcher:setDisp12MaxDiff(t)
        local argRules = {
            {"disp12MaxDiff", required = true}}
        local disp12MaxDiff = cv.argcheck(t, argRules)
        C.StereoMatcher_setDisp12MaxDiff(self.ptr, disp12MaxDiff)
    end

    function StereoMatcher:setMinDisparity(t)
        local argRules = {
            {"minDisparity", required = true}}
        local minDisparity = cv.argcheck(t, argRules)
        C.StereoMatcher_setMinDisparity(self.ptr, minDisparity)
    end

    function StereoMatcher:setNumDisparities(t)
        local argRules = {
            {"numDisparities", required = true}}
        local numDisparities = cv.argcheck(t, argRules)
        C.StereoMatcher_setNumDisparities(self.ptr, numDisparities)
    end

    function StereoMatcher:setSpeckleRange(t)
        local argRules = {
            {"speckleRange", required = true}}
        local speckleRange = cv.argcheck(t, argRules)
        C.StereoMatcher_setSpeckleRange(self.ptr, speckleRange)
    end

    function StereoMatcher:setSpeckleWindowSize(t)
        local argRules = {
            {"speckleWindowSize", required = true}}
        local speckleWindowSize = cv.argcheck(t, argRules)
        C.StereoMatcher_setSpeckleWindowSize(self.ptr, speckleWindowSize)
    end
end

--StereoBM

do
    local StereoBM = torch.class('cv.StereoBM', 'cv.StereoMatcher', cv)

    function StereoBM:__init()
        local argRules = {
            {"numDisparities", default = 0},
            {"blockSize", default = 21}}
        local numDisparities, blockSize = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(
			C.StereoBM_ctor(numDisparities, blockSize),
			Classes.Algorithm_dtor)
    end

    function StereoBM:getPreFilterCap()
        return C.StereoBM_getPreFilterCap(self.ptr)
    end

    function StereoBM:getPreFilterSize()
        return C.StereoBM_getPreFilterSize(self.ptr)
    end

    function StereoBM:getPreFilterType()
        return C.StereoBM_getPreFilterType(self.ptr)
    end

    function StereoBM:getROI1()
        return C.StereoBM_getROI1(self.ptr)
    end

    function StereoBM:getROI2()
        return C.StereoBM_getROI2(self.ptr)
    end

    function StereoBM:getSmallerBlockSize()
        return C.StereoBM_getSmallerBlockSize(self.ptr)
    end

    function StereoBM:getTextureThreshold()
        return C.StereoBM_getTextureThreshold(self.ptr)
    end

    function StereoBM:getUniquenessRatio()
        return C.StereoBM_getUniquenessRatio(self.ptr)
    end

    function StereoBM:setPreFilterCap()
        local argRules = {
            {"preFilterCap", required = true}}
        local preFilterCap = cv.argcheck(t, argRules)
        C.StereoBM_setPreFilterCap(self.ptr, preFilterCap)
    end

    function StereoBM:setPreFilterSize()
        local argRules = {
            {"preFilterSize", required = true}}
        local preFilterSize = cv.argcheck(t, argRules)
        C.StereoBM_setPreFilterSize(self.ptr, preFilterSize)
    end

    function StereoBM:setPreFilterType()
        local argRules = {
            {"preFilterType", required = true}}
        local preFilterType = cv.argcheck(t, argRules)
        C.StereoBM_setPreFilterType(self.ptr, preFilterType)
    end

    function StereoBM:setROI1()
        local argRules = {
            {"roi1", required = true}}
        local roi1 = cv.argcheck(t, argRules)
        C.StereoBM_setROI1(self.ptr, roi1)
    end

    function StereoBM:setROI2()
        local argRules = {
            {"roi2", required = true}}
        local roi2 = cv.argcheck(t, argRules)
        C.StereoBM_setROI2(self.ptr, roi2)
    end

    function StereoBM:setSmallerBlockSize()
        local argRules = {
            {"blockSize", required = true}}
        local blockSize = cv.argcheck(t, argRules)
        C.StereoBM_setSmallerBlockSize(self.ptr, blockSize)
    end

    function StereoBM:setTextureThreshold()
        local argRules = {
            {"textureThreshold", required = true}}
        local textureThreshold = cv.argcheck(t, argRules)
        C.StereoBM_setTextureThreshold(self.ptr, textureThreshold)
    end

    function StereoBM:setUniquenessRatio()
        local argRules = {
            {"uniquenessRatio", required = true}}
        local uniquenessRatio = cv.argcheck(t, argRules)
        C.StereoBM_setUniquenessRatio(self.ptr, uniquenessRatio)
    end
end

--StereoSGBM

do
    local StereoSGBM = torch.class('cv.StereoSGBM', 'cv.StereoMatcher', cv)

    function StereoSGBM:__init()
        local argRules = {
            {"minDisparity", required = true},
            {"numDisparities", required = true},
            {"blockSize", required = true},
            {"P1", default = 0},
            {"P2", default = 0},
            {"disp12MaxDiff", default = 0},
            {"preFilterCap", default = 0},
            {"uniquenessRatio", default = 0},
            {"speckleWindowSize", default = 0},
            {"speckleRange", default = 0},
            {"mode", default = cv.StereoSGBM_MODE_SGBM}}
        local minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff,
              preFilterCap, uniquenessRatio, speckleWindowSize,
              speckleRange, mode = cv.argcheck(t, argRules)
        self.ptr = ffi.gc(
			C.StereoSGBM_ctor(
				minDisparity, numDisparities, blockSize, P1, P2,
				disp12MaxDiff, preFilterCap, uniquenessRatio,
				speckleWindowSize, speckleRange, mode), Classes.Algorithm_dtor)
    end

    function StereoSGBM:getMode()
        return C.StereoSGBM_getMode(self.ptr)
    end

    function StereoSGBM:getP1()
        return C.StereoSGBM_getP1(self.ptr)
    end

    function StereoSGBM:getP2()
        return C.StereoSGBM_getP2(self.ptr)
    end

    function StereoSGBM:getPreFilterCap()
        return C.StereoSGBM_getPreFilterCap(self.ptr)
    end

    function StereoSGBM:getUniquenessRatio()
        return C.StereoSGBM_getUniquenessRatio(self.ptr)
    end

    function StereoSGBM:setMode(t)
        local argRules = {
            {"mode", required = true}}
        local mode = cv.argcheck(t, argRules)
        C.StereoSGBM_setMode(self.ptr, mode)
    end

    function StereoSGBM:setP1(t)
        local argRules = {
            {"P1", required = true}}
        local P1 = cv.argcheck(t, argRules)
        C.StereoSGBM_setP1(self.ptr, P1)
    end

    function StereoSGBM:setP2(t)
        local argRules = {
            {"P2", required = true}}
        local P2 = cv.argcheck(t, argRules)
        C.StereoSGBM_setP2(self.ptr, P2)
    end

    function StereoSGBM:setPreFilterCap(t)
        local argRules = {
            {"preFilterCap", required = true}}
        local preFilterCap = cv.argcheck(t, argRules)
        C.StereoSGBM_setPreFilterCap(self.ptr, preFilterCap)
    end

    function StereoSGBM:setUniquenessRatio(t)
        local argRules = {
            {"uniquenessRatio", required = true}}
        local uniquenessRatio = cv.argcheck(t, argRules)
        C.StereoSGBM_setUniquenessRatio(self.ptr, uniquenessRatio)
    end
end

return cv










