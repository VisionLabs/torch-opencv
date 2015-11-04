require 'cv'

local ffi = require 'ffi'

ffi.cdef[[
struct PtrWrapper {
    void *ptr;
};
]]

C = ffi.load(libPath('Classes'))

-- ***** FileNode *****

ffi.cdef[[
struct PtrWrapper FileNode_ctor();

void FileNode_dtor(struct PtrWrapper ptr);
]]

do
	local FileNode = torch.class('cv.FileNode')

	function FileNode:__init()
		self.ptr = ffi.gc(C.FileNode_ctor(), C.FileNode_dtor)
	end
end

-- ***** FileStorage *****

ffi.cdef[[
struct PtrWrapper FileStorage_ctor(const char *source, int flags, const char *encoding);

struct PtrWrapper FileStorage_ctor_default();

void FileStorage_dtor(struct PtrWrapper ptr);

bool FileStorage_open(struct PtrWrapper ptr, const char *filename, int flags, const char *encoding);

bool FileStorage_isOpened(struct PtrWrapper ptr);

void FileStorage_release(struct PtrWrapper ptr);

const char *FileStorage_releaseAndGetString(struct PtrWrapper ptr);
]]

do
    local FileStorage = torch.class('cv.FileStorage')

    function FileStorage:__init(t)
        local source = t.source
        local flags = t.flags
        local encoding = t.encoding or ''

        if source and flags then
	        self.ptr = ffi.gc(C.FileStorage_ctor(source, flags, encoding), C.FileStorage_dtor)
	    else
	    	self.ptr = ffi.gc(C.FileStorage_ctor_default(), C.FileStorage_dtor)
	    end
    end

    function FileStorage:open(t)
    	local source = assert(t.source)
        local flags = assert(t.flags)
        local encoding = t.encoding or ''

        return C.FileStorage_open(self.ptr, source, flags, encoding)
    end

    function FileStorage:isOpened()
    	return C.FileStorage_isOpened(self.ptr)
    end

    function FileStorage:release()
    	C.FileStorage_release(self.ptr)
    end

    function FileStorage:releaseAndGetString()
    	return ffi.string(C.FileStorage_releaseAndGetString(self.ptr))
    end
end

-- ***** Algorithm *****

ffi.cdef[[
struct PtrWrapper Algorithm_ctor();

void Algorithm_dtor(struct PtrWrapper ptr);

void Algorithm_clear(struct PtrWrapper ptr);

void Algorithm_write(struct PtrWrapper ptr, struct PtrWrapper fileStorage);

void Algorithm_read(struct PtrWrapper ptr, struct PtrWrapper fileNode);

bool Algorithm_empty(struct PtrWrapper ptr);

void Algorithm_save(struct PtrWrapper ptr, const char *filename);

const char *Algorithm_getDefaultName(struct PtrWrapper ptr);
]]

do
    local Algorithm = torch.class('cv.Algorithm')

    function Algorithm:__init()
        self.ptr = ffi.gc(C.Algorithm_ctor(), C.Algorithm_dtor)
    end

    function Algorithm:clear()
        C.Algorithm_clear(self.ptr)
    end

    function Algorithm:write(fileStorage)
    	C.Algorithm_write(self.ptr, fileStorage.ptr)
    end

    function Algorithm:read(fileNode)
    	C.Algorithm_read(self.ptr, fileNode.ptr)
    end

    function Algorithm:empty()
    	return C.Algorithm_empty(self.ptr)
    end

    function Algorithm:save(filename)
    	C.Algorithm_save(self.ptr, filename)
    end

    function Algorithm:getDefaultName()
    	return C.Algorithm_getDefaultName(self.ptr)
    end
end