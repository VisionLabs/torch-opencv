PROJECT(Torch_OpenCV)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)

OPTION(BUILD_TESTS   "Build C tests"                OFF)
OPTION(BUILD_CUDA    "Wrap CUDA packages"           ON)

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

SET(OPENCV_MODULES_TO_WRAP
        core imgcodecs highgui features2d flann imgproc ml optflow photo superres videoio video
        objdetect calib3d ximgproc stitching xphoto)

IF (BUILD_CUDA)
    SET(OPENCV_MODULES_TO_WRAP ${OPENCV_MODULES_TO_WRAP} cudaarithm cudacodec cudawarping
        cudafilters cudaoptflow cudabgsegm cudafeatures2d cudaimgproc cudastereo cudaobjdetect)
ENDIF()

IF (DEFINED ENV{DONT_WRAP})
    SET(DONT_WRAP $ENV{DONT_WRAP})
ENDIF()

IF (DONT_WRAP)
    LIST(REMOVE_ITEM OPENCV_MODULES_TO_WRAP ${DONT_WRAP})
ENDIF()

# Unfortunately, OpenCVConfig.cmake doesn't make use of FIND_PACKAGE's "OPTIONAL_COMPONENTS"
# So let's invoke FIND_PACKAGE twice:

# Once (error-throwing) to check if OpenCV is present
FIND_PACKAGE(OpenCV 3.1 EXACT REQUIRED)
# Once more (silently) to look for the installed modules
FIND_PACKAGE(OpenCV 3.1 EXACT QUIET OPTIONAL_COMPONENTS ${OPENCV_MODULES_TO_WRAP})

IF(LUAROCKS_PREFIX)
    MESSAGE(STATUS "Installing Torch through Luarocks")
    STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
    MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()

FIND_PACKAGE(Torch REQUIRED)
INCLUDE_DIRECTORIES(${Torch_INSTALL_INCLUDE} ${OpenCV_INCLUDE_DIRS})

# See #78
IF (APPLE)
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
ENDIF()

SET(CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_STANDARD 11)

# See #90
IF (CMAKE_VERSION VERSION_LESS 3.1)
    SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
ENDIF()

FILE(GLOB luasrc *.lua cv/*.lua)
SET(luasrc ${luasrc})
ADD_TORCH_PACKAGE(cv "" "${luasrc}" "cv")

# Check if both OpenCV CUDA interface and cutorch are installed
IF (BUILD_CUDA)
    IF (NOT OPENCV_CUDAARITHM_FOUND)
        MESSAGE(WARNING "BUILD_CUDA is ON, but OpenCV CUDA interface wasn't found. Won't build OpenCV-CUDA wrappers.")
        SET(BUILD_CUDA OFF)
    ELSEIF(NOT EXISTS "${Torch_INSTALL_INCLUDE}/THC/")
        MESSAGE(WARNING "BUILD_CUDA is ON, but cutorch wasn't found. Won't build OpenCV-CUDA wrappers")
        SET(BUILD_CUDA OFF)
    ENDIF()
ENDIF()

IF (BUILD_CUDA)
    FIND_PACKAGE(CUDA REQUIRED)

    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    ADD_DEFINITIONS(-DWITH_CUDA)
ENDIF()

SET(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/include")

ADD_LIBRARY(Common SHARED "${SOURCE_DIR}/Common.cpp")
TARGET_LINK_LIBRARIES(Common TH ${OpenCV_LIBS})

ADD_LIBRARY(Classes SHARED "${SOURCE_DIR}/Classes.cpp")
TARGET_LINK_LIBRARIES(Classes Common)

IF (BUILD_CUDA)
    ADD_LIBRARY(CUDACommon SHARED "${SOURCE_DIR}/CUDACommon.cpp")
    TARGET_LINK_LIBRARIES(CUDACommon Common THC ${CUDA_LIBRARIES})
ENDIF()

# The main loop. For each module check if it's present; if yes, add its wrappers to targets

FOREACH(MODULE ${OPENCV_MODULES_TO_WRAP})
    STRING(TOUPPER ${MODULE} MODULE_UPPERCASE)
    IF (OPENCV_${MODULE_UPPERCASE}_FOUND)
        IF (${MODULE_UPPERCASE} MATCHES "^CUDA")
            IF (BUILD_CUDA)
                ADD_SUBDIRECTORY(cv/${MODULE})
                ADD_LIBRARY(${MODULE} SHARED "${SOURCE_DIR}/${MODULE}.cpp")
                TARGET_LINK_LIBRARIES(${MODULE} CUDACommon)
            ENDIF()
        ELSE()
            ADD_SUBDIRECTORY(cv/${MODULE})
            ADD_LIBRARY(${MODULE} SHARED "${SOURCE_DIR}/${MODULE}.cpp")
            TARGET_LINK_LIBRARIES(${MODULE} Common)
        ENDIF()

        SET(RESULTING_MODULES_LIST "${MODULE}, ${RESULTING_MODULES_LIST}")
    ENDIF()
ENDFOREACH()

MESSAGE(STATUS "Modules to wrap: ${RESULTING_MODULES_LIST}")

INSTALL(DIRECTORY "${CMAKE_BINARY_DIR}/lib" DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/cv")

IF (BUILD_TESTS)
    FILE(GLOB Tests_SRC "${SOURCE_DIR}/tests/*.cpp")
    ADD_LIBRARY(Tests SHARED ${Tests_SRC})
    TARGET_LINK_LIBRARIES(Tests Common)
ENDIF()
