PROJECT(Torch_OpenCV)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)

OPTION(BUILD_TESTS      "Build C tests"         OFF)
OPTION(BUILD_CUDA_CV    "Build CUDA packages"   ON)

SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

SET(OPENCV_MODULES_TO_WRAP
        imgcodecs highgui features2d flann imgproc ml optflow photo superres videoio
        cudaarithm)

# Unfortunately, OpenCVConfig.cmake doesn't make use of FIND_PACKAGE's "OPTIONAL_COMPONENTS"
# So let's invoke FIND_PACKAGE twice:

# Once (error-throwing) to check if OpenCV is present
FIND_PACKAGE(OpenCV 3 REQUIRED)
# Once more (silently) to look for the installed modules
FIND_PACKAGE(OpenCV 3 OPTIONAL_COMPONENTS ${OPENCV_MODULES_TO_WRAP})

IF(LUAROCKS_PREFIX)
    MESSAGE(STATUS "Installing Torch through Luarocks")
    STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
    MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()
FIND_PACKAGE(Torch REQUIRED)
INCLUDE_DIRECTORIES(${Torch_INSTALL_INCLUDE} ${OpenCV_INCLUDE_DIRS})

FILE(GLOB luasrc *.lua cv/*.lua)
SET(luasrc ${luasrc})
ADD_TORCH_PACKAGE(cv "" "${luasrc}" "cv")

SET(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/include")

ADD_LIBRARY(Common SHARED "${SOURCE_DIR}/Common.cpp")
TARGET_LINK_LIBRARIES(Common TH ${OpenCV_LIBS})

ADD_LIBRARY(Classes SHARED "${SOURCE_DIR}/Classes.cpp")
TARGET_LINK_LIBRARIES(Classes Common)

# Check if both OpenCV CUDA interface and cutorch are installed
IF (BUILD_CUDA_CV)
    IF (NOT OPENCV_CUDAARITHM_FOUND)
        MESSAGE(WARNING "BUILD_CUDA_CV is ON, but OpenCV CUDA interface wasn't found. Won't build OpenCV-CUDA wrappers")
        SET(BUILD_CUDA_CV OFF)
    ELSEIF(NOT EXISTS "${Torch_INSTALL_INCLUDE}/THC/")
        MESSAGE(WARNING "BUILD_CUDA_CV is ON, but cutorch wasn't found. Won't build OpenCV-CUDA wrappers")
        SET(BUILD_CUDA_CV OFF)
    ENDIF()
ENDIF()

IF (BUILD_CUDA_CV)
    FIND_PACKAGE(CUDA REQUIRED)

    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

    ADD_LIBRARY(CUDACommon SHARED "${SOURCE_DIR}/CUDACommon.cpp")
    TARGET_LINK_LIBRARIES(CUDACommon Common)
ENDIF()

FOREACH(MODULE ${OPENCV_MODULES_TO_WRAP})
    ADD_SUBDIRECTORY(cv/${MODULE})
    STRING(TOUPPER ${MODULE} MODULE_UPPERCASE)
    IF (OPENCV_${MODULE_UPPERCASE}_FOUND)
        IF (${MODULE_UPPERCASE} MATCHES "^CUDA")
            IF (BUILD_CUDA_CV)
                ADD_LIBRARY(${MODULE} SHARED "${SOURCE_DIR}/${MODULE}.cpp")
                TARGET_LINK_LIBRARIES(${MODULE} CUDACommon)
            ENDIF()
        ELSE()
            ADD_LIBRARY(${MODULE} SHARED "${SOURCE_DIR}/${MODULE}.cpp")
            TARGET_LINK_LIBRARIES(${MODULE} Common)
        ENDIF()
    ENDIF()
ENDFOREACH()

INSTALL(DIRECTORY "${CMAKE_BINARY_DIR}/lib" DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/cv")

IF (BUILD_TESTS)
    FILE(GLOB Tests_SRC "${SOURCE_DIR}/tests/*.cpp")
    ADD_LIBRARY(Tests SHARED ${Tests_SRC})
    TARGET_LINK_LIBRARIES(Tests Common)
endif()



