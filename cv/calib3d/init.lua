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

extern "C"
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
        {"criteria", default = 0, operator = cv.TermCriteria}}
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
    return retval[1][1], retval[2][1], retval[3][1], cv.Point2d(retval[4][1], retval[5][1]),retval[6][1]
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


--TODO  don't work cv.wrap_tensors 
function cv.test(t)
    local argRules = {
        {"src", required = true}}
    local src = cv.argcheck(t,argRules)
    cv.wrap_tensors(src)
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
    return cv.unwrap_tensors(C.findChessboardCorners(cv.wrap_tensor(image), patternSize, flags))
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

return cv















