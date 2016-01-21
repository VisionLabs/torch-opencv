local cv = require 'cv._env'
local ffi = require 'ffi'

ffi.cdef[[
double calibrateCamera(
	struct TensorArray objectPoints, struct TensorArray imagePoints,
	struct SizeWrapper imageSize, struct TensorWrapper cameraMatrix,
	struct TensorWrapper distCoeffs, struct TensorArray rvecs,
	struct TensorArray tvecs, int flags, struct TermCriteriaWrapper criteria);

struct TensorWrapper calibrationMatrixValues(
	struct TensorWrapper cameraMatrix,
	struct SizeWrapper imageSize,
	double apertureWidth, double apertureHeight);

void composeRT(
	struct TensorWrapper rvec1, struct TensorWrapper tvec1, struct TensorWrapper rvec2,
	struct TensorWrapper tvec2, struct TensorWrapper rvec3, struct TensorWrapper tvec3,
	struct TensorWrapper dr3dr1, struct TensorWrapper dr3dt1, struct TensorWrapper dr3dr2,
	struct TensorWrapper dr3dt2, struct TensorWrapper dt3dr1, struct TensorWrapper dt3dt1,
	struct TensorWrapper dt3dr2, struct TensorWrapper dt3dt2);

struct TensorWrapper computeCorrespondEpilines(
	struct TensorWrapper points, int whichImage, struct TensorWrapper F);

struct TensorWrapper convertPointsFromHomogeneous(
	struct TensorWrapper src);

struct TensorWrapper convertPointsHomogeneous(
	struct TensorWrapper src, struct TensorWrapper dst);

struct TensorWrapper convertPointsToHomogeneous(
	struct TensorWrapper src);

struct TensorArray correctMatches(
	struct TensorWrapper F, struct TensorWrapper points1,
	struct TensorWrapper points2);

struct TensorArray decomposeEssentialMat(
	struct TensorWrapper E);

struct TensorArrayPlusInt decomposeHomographyMat(
	struct TensorWrapper H, struct TensorWrapper K);

struct TensorArray decomposeProjectionMatrix(
	struct TensorWrapper projMatrix, struct TensorWrapper rotMatrixX,
	struct TensorWrapper rotMatrixY, struct TensorWrapper rotMatrixZ,
	struct TensorWrapper eulerAngles);

void drawChessboardCorners(
	struct TensorWrapper image, struct SizeWrapper patternSize,
	struct TensorWrapper corners, bool patternWasFound);

struct TensorArrayPlusInt estimateAffine3D(
	struct TensorWrapper src, struct TensorWrapper dst,
	double ransacThreshold, double confidence);

void filterSpeckles(
	struct TensorWrapper img, double newVal, int maxSpeckleSize,
	double maxDiff, struct TensorWrapper buf);
 
void find4QuadCornerSubpix(
	struct TensorWrapper img, struct TensorWrapper corners,
	struct SizeWrapper region_size);

struct TensorWrapper findChessboardCorners(
	struct TensorWrapper image, struct SizeWrapper patternSize, int flags);

//TODO const Ptr<FeatureDetector>& blobDetector = SimpleBlobDetector::create()
struct TensorPlusBool findCirclesGrid(
	struct TensorWrapper image, struct SizeWrapper patternSize, int flags);

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
	struct TensorWrapper A, struct TensorWrapper B);

struct TensorArray projectPoints(
	struct TensorWrapper objectPoints, struct TensorWrapper rvec,
	struct TensorWrapper tvec, struct TensorWrapper cameraMatrix,
	struct TensorWrapper distCoeffs, struct TensorWrapper imagePoints,
	struct TensorWrapper jacobian, double aspectRatio);

struct TensorArrayPlusInt recoverPose(
	struct TensorWrapper E, struct TensorWrapper points1,
	struct TensorWrapper points2, double focal,
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

double stereoCalibrate(
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

]]

local C = ffi.load(cv.libPath('calib3d'))

function cv.calibrateCamera(t)
    local argRules = {
        {"objectPoints", required = true},
        {"imagePoints", required = true},
        {"imageSize", required = true, operator = cv.Size},
        {"cameraMatrix", required = true}, 
        {"distCoeffs", required = true},
        {"rvecs", required = true},
        {"tvecs", required = true},
        {"flag", default = 0},
        {"criteria", default = cv.TermCriteria(TERM_CRITERIA_COUNT+TERM_CRITERIA_EPS),
                               operator = cv.TermCriteria}}
    local objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs,
      		rvecs, tvecs, flag, criteria = cv.argcheck(t, argRules)
    local result = C.calibrateCamera(
			cv.wrap_tensors(objectPoints), cv.wrap_tensors(imagePoints),
			imageSize, cv.wrap_tensor(cameraMatrix), cv.wrap_tensor(distCoeffs),
			cv.wrap_tensors(rvecs), cv.wrap_tensors(tvecs), flag, criteria)
    return result
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
           cv.Point2d(retval[4][1], retval[5][1]),retval[6][1]
end

function cv.composeRT(t)
    local argRules = {
        {"rvec1", required = true},
        {"tvec1", required = true},
        {"rvec2", required = true},
        {"tvec2", required = true},
        {"rvec3", required = true},
        {"tvec3", required = true},
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
    C.composeRT(
	rvec1, tvec1, rvec2, tvec2, rvec3, tvec3, dr3dr1, dr3dt1,
	dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2)
end   

function cv.computeCorrespondEpilines(t)
    local argRules = {
        {"points", required = true},
        {"whichImage", required = true},
        {"F", required = true}}
    local points, whichImage, F = cv.argcheck(t, argRules)
    local result = C.computeCorrespondEpilines(
				cv.wrap_tensor(points), whichImage, cv.wrap_tensor(F))
    return cv.unwrap_tensors(result)
end

function cv.convertPointsFromHomogeneous(t)
    local argRules = {
        {"src", required = true}}
    local src = cv.argcheck(t, argRules)
    local result = C.convertPointsFromHomogeneous(cv.wrap_tensor(src))
    return cv.unwrap_tensors(result)
end

function cv.convertPointsHomogeneous(t)
    local argRules = {
        {"src", required = true},
        {"dst", required = true}}
    local src, dst = cv.argcheck(t, argRules)
    C.convertPointsHomogeneous(cv.wrap_tensor(src), cv.wrap_tensor(dst))
end

function cv.convertPointsToHomogeneous(t)
    local argRules = {
        {"src", required = true}}
    local src = cv.argcheck(t, argRules)
    local result = C.convertPointsToHomogeneous(cv.wrap_tensor(src))
    return cv.unwrap_tensors(result)
end

function cv.correctMatches(t)
    local argRules = {
        {"F",required = true},
        {"points1", required = true},
        {"points2", required = true}}
    local F, points1, points2 = cv.argcheck(t, argRules)
    local result = C.correctMatches(
			cv.wrap_tensor(F), cv.wrap_tensor(points1), cv.wrap_tensor(points2))
    return unwrap_tensors(result)
end

function cv.decomposeEssentialMat(t)
    local argRules = {
        {"E", required = true}}
    local E = cv.argcheck(t, argRules)
    local result = C.decomposeEssentialMat(E)
    return unwrap_tensors(result)
end

function cv.decomposeHomographyMat(t)
    local argRules = {
       {"H", required = true},
       {"K", required = true}}
    local H, K = cv.argcheck(t, argRules)
    local result = C.decomposeHomographyMat(cv.wrap_tensor(H), cv.wrap_tensor(K))
    return unwrap_tensors(result)
end

function cv.decomposeProjectionMatrix(t)
    local argRules = {
        {"projMatrix", required = true},
        {"rotMatrixX", default = nil},
        {"rotMatrixY", default = nil},
        {"rotMatrixZ", default = nil},
        {"eulerAngles", default = nil}}
    local projMatrix, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv.argcheck(t, argRules)
    local result = C.decomposeProjectionMatrix(
			cv.wrap_tensor(projMatrix), cv.wrap_tensor(rotMatrixX),
			cv.wrap_tensor(rotMatrixY), cv.wrap_tensor(rotMatrixZ),
			cv.wrap_tensor(eulerAngles))
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
        {"ransacThreshold", default = 3},
        {"confidence", default = 0.99}}
    local src, dst, ransacThreshold, confidence = cv.argcheck(t, argRules)
    local result = C.estimateAffine3D(
				cv.wrap_tensor(src), cv.wrap_tensor(dst),
				ransacThreshold, confidence)
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
        {"flags", default = CALIB_CB_ADAPTIVE_THRESH+CALIB_CB_NORMALIZE_IMAGE}}
    local image, patternSize, flags = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(
		C.findChessboardCorners(cv.wrap_tensor(image), patternSize, flags))
end

--TODO const Ptr<FeatureDetector>& blobDetector = SimpleBlobDetector::create()
function cv.findCirclesGrid(t)
    local argRules = {
        {"image", required = true},
        {"patternSize", required = true, operator = cv.Size},
        {"flags", default = CALIB_CB_SYMMETRIC_GRID}}
    local image, patternSize, flags = cv.argcheck(t, argRules)
    local result = C.findCirclesGrid(cv.wrap_tensor(image), patternSize, flags)
    return result.val, cv.unwrap_tensors(result.tensor)
end 

function cv.findEssentialMat(t)
    local argRules = {
        {"points1", required = true},
        {"points2", required = true},
        {"focal", default = 1.0},
        {"pp", operator = cv.Point2d, default = cv.Point2d(0,0)},
        {"method", default = RANSAC},
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
        {"method", default = FM_RANSAC},
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
	{"mask", required = true},
	{"method", default = FM_RANSAC},
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
        {"B", required = true}}
    local A, B = cv.argcheck(t, argRules)
    return cv.unwrap_tensors(C.matMulDeriv(A, B))
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
        {"focal", default = 1.0},
        {"Point2d", default = cv.Point2d(0,0), operator = cv.Point2d},
        {"mask", default = nil}}
    local E, points1, points2, focal, Point2d, mask = cv.argcheck(t, argRules)
    local result = C.recoverPose(
			cv.wrap_tensor(E), cv.wrap_tensor(points1),
			cv.wrap_tensor(points2), focal, Point2d, mask)
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
        {"flags", default = SOLVEPNP_ITERATIVE}}
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
        {"flags", default = SOLVEPNP_ITERATIVE}}
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
        {"cameraMatrix1", required = true},
        {"distCoeffs1", required = true},
        {"cameraMatrix2", required = true},
        {"distCoeffs2", required = true},
        {"imageSize", required = true, operator = cv.Size},
        {"R", required = true},
        {"T", required = true},
        {"E", required = true},
        {"F", required = true},
        {"flags", defauly = CALIB_FIX_INTRINSIC},
        {"criteria", default = cv.TermCriteria(TERM_CRITERIA_COUNT+TERM_CRITERIA_EPS),
                               operator = cv.TermCriteria}}
    local objectPoints, imagePoints1, imagePoints2, cameraMatrix1,
          distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T,
          E, F, flags, criteria = cv.argcheck(t, argRules)
    return C.stereoCalibrate(
		cv.wrap_tensor(objectPoints), cv.wrap_tensor(imagePoints1),
		cv.wrap_tensor(imagePoints2), cv.wrap_tensor(cameraMatrix1),
		cv.wrap_tensor(distCoeffs1), cv.wrap_tensor(cameraMatrix2),
		cv.wrap_tensor(distCoeffs2), imageSize, cv.wrap_tensor(R),
		cv.wrap_tensor(T), cv.wrap_tensor(E), cv.wrap_tensor(F),
		flags, criteria)
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
        {"flags", default = CALIB_ZERO_DISPARITY},
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


return cv















