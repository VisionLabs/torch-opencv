require 'cv'

local ffi = require 'ffi'

local C = ffi.load(libPath('videoio'))

--- ***************** Classes *****************
require 'cv.Classes'

-- VideoCapture

ffi.cdef[[
struct PtrWrapper VideoCapture_ctor_default();

struct PtrWrapper VideoCapture_ctor_device(int device);

struct PtrWrapper VideoCapture_ctor_filename(const char *filename);

void VideoCapture_dtor(struct PtrWrapper ptr);

bool VideoCapture_open(struct PtrWrapper ptr, int device);

bool VideoCapture_isOpened(struct PtrWrapper ptr);

void VideoCapture_release(struct PtrWrapper ptr);

bool VideoCapture_grab(struct PtrWrapper ptr);

struct TensorPlusBool VideoCapture_retrieve(
        struct PtrWrapper ptr, struct TensorWrapper image, int flag);

struct TensorPlusBool VideoCapture_read(
        struct PtrWrapper ptr, struct TensorWrapper image);

bool VideoCapture_set(struct PtrWrapper ptr, int propId, double value);

double VideoCapture_get(struct PtrWrapper ptr, int propId);
]]

do
	local VideoCapture = torch.class('cv.VideoCapture')

	-- v = cv.VideoCapture{}
	-- OR
	-- v = cv.VideoCapture{filename='../video.mp4'}
	-- OR
	-- v = cv.VideoCapture{device=2}
	function VideoCapture:__init(t)
		if t.filename then
			self.ptr = ffi.gc(C.VideoCapture_ctor_filename(t.filename), C.VideoCapture_dtor)
		elseif t.device then
			self.ptr = ffi.gc(C.VideoCapture_ctor_device(t.device), C.VideoCapture_dtor)
		else
			self.ptr = ffi.gc(C.VideoCapture_ctor_default(), C.VideoCapture_dtor)
		end
	end

	function VideoCapture:open(t)
		local device = assert(t.device)
		
		return C.VideoCapture_open(self.ptr, device)
	end

	function VideoCapture:isOpened()
		return C.VideoCapture_isOpened(self.ptr)
	end

	function VideoCapture:release()
		C.VideoCapture_release(self.ptr)		
	end

	function VideoCapture:grab()
		return C.VideoCapture_grab(self.ptr)
	end

	function VideoCapture:retrieve(t)
		local image = t.image
		local flag = t.flag or 0

		result = C.VideoCapture_retrieve(self.ptr, cv.wrap_tensors(image), flag)
		return result.val, cv.unwrap_tensors(result.tensor)
	end

	-- result, image = cap.read{}
	-- OR
	-- im = torch.FloatTensor(640, 480, 3)
	-- result = cap.read{image=image}
	function VideoCapture:read(t)
		local image = t.image

		result = C.VideoCapture_read(self.ptr, cv.wrap_tensors(image))
		return result.val, cv.unwrap_tensors(result.tensor)
	end

	function VideoCapture:set(propId, value)
		return C.VideoCapture_set(self.ptr, propId, value)
	end

	function VideoCapture:get(propId)
		return C.VideoCapture_get(self.ptr, propId)
	end
end

-- VideoWriter

ffi.cdef[[
struct PtrWrapper VideoWriter_ctor_default();

struct PtrWrapper VideoWriter_ctor(
        const char *filename, int fourcc, double fps, struct SizeWrapper frameSize, bool isColor);

void VideoWriter_dtor(struct PtrWrapper ptr);

bool VideoWriter_open(struct PtrWrapper ptr, const char *filename, int fourcc, 
                      double fps, struct SizeWrapper frameSize, bool isColor);

bool VideoWriter_isOpened(struct PtrWrapper ptr);

void VideoWriter_release(struct PtrWrapper ptr);

void VideoWriter_write(struct PtrWrapper ptr, struct TensorWrapper image);

bool VideoWriter_set(struct PtrWrapper ptr, int propId, double value);

double VideoWriter_get(struct PtrWrapper ptr, int propId);

int VideoWriter_fourcc(char c1, char c2, char c3, char c4);
]]

do
	local VideoWriter = torch.class('cv.VideoWriter')

	function VideoWriter:__init(t)
		if t.filename then
			local filename = assert(t.filename)
			local fourcc = assert(t.fourcc)
			local fps = assert(t.fps)
			local frameSize = cv.Size(assert(t.frameSize))
			local isColor = t.isColor
			if isColor == nil then isColor = true end
			self.ptr = ffi.gc(C.VideoWriter_ctor(
				filename, fourcc, fps, frameSize, isColor), 
				C.VideoWriter_dtor)
		else
			self.ptr = ffi.gc(C.VideoWriter_ctor_default(), C.VideoWriter_dtor)
		end
	end

	function VideoWriter:open(t)
		local filename = assert(t.filename)
		local fourcc = assert(t.fourcc)
		local fps = assert(t.fps)
		local frameSize = cv.Size(assert(t.frameSize))
		local isColor = t.isColor
		if isColor == nil then isColor = true end
		
		return C.VideoWriter_open(self.ptr, filename, fourcc, fps, frameSize, isColor)
	end

	function VideoWriter:isOpened()
		return C.VideoWriter_isOpened(self.ptr)
	end

	function VideoWriter:release()
		C.VideoWriter_release(self.ptr)		
	end

	function VideoWriter:write(t)
		local image = assert(t.image)
		C.VideoWriter_write(self.ptr, image)
	end

	function VideoWriter:set(propId, value)
		return C.VideoWriter_set(self.ptr, propId, value)
	end

	function VideoWriter:get(propId)
		return C.VideoWriter_get(self.ptr, propId)
	end
end