require 'cv'

local ffi = require 'ffi'

local C = ffi.load(libPath('videoio'))

--- ***************** Classes *****************
require 'cv.Classes'

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