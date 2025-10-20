cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

project(externals)

externalproject_add(
    openblas
    URL               ${CMAKE_SOURCE_DIR}/externals/OpenBLAS-0.3.21.tar.gz
    PREFIX            ${CMAKE_BINARY_DIR}/openblas
    INSTALL_DIR       ${CMAKE_INSTALL_PREFIX}
    
    TEST_COMMAND      ""

    CMAKE_ARGS       -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                     -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                     -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                     -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
                     -DCMAKE_BUILD_TYPE=Release
                     -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                     -DCMAKE_POLICY_VERSION_MINIMUM=3.5
                     -DBUILD_SHARED_LIBS=OFF
                     -DBUILD_WITHOUT_LAPACK=ON
                     -DDYNAMIC_ARCH=ON
                     -DUSE_THREAD=ON
                     -DUSE_OPENMP=OFF
                     -DNOFORTRAN=ON
                     -DBUILD_SINGLE=ON # Float support. Remove if need double and complex types too.

    LOG_DOWNLOAD          TRUE
    LOG_UPDATE            TRUE
    LOG_CONFIGURE         TRUE
    LOG_BUILD             TRUE
    LOG_INSTALL           TRUE
    LOG_MERGED_STDOUTERR  TRUE
    LOG_OUTPUT_ON_FAILURE TRUE

    DOWNLOAD_EXTRACT_TIMESTAMP ON
)

externalproject_add(
    opencv
    URL               ${CMAKE_SOURCE_DIR}/externals/opencv-4.1.0.zip
    PREFIX            ${CMAKE_BINARY_DIR}/opencv
    INSTALL_DIR       ${CMAKE_INSTALL_PREFIX}

    TEST_COMMAND      ""

    LIST_SEPARATOR   ","
    CMAKE_ARGS       -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                     -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
                     -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                     -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                     -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                     -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=ON
                     -DOPENCV_ENABLE_NONFREE=OFF
                     -DOPENCV_FORCE_3RDPARTY_BUILD=OFF
                     -DBUILD_LIST=core,imgproc,calib3d,highgui,objdetect,videoio
                     -DBUILD_SHARED_LIBS=OFF
                     -DBUILD_TESTS=OFF
                     -DBUILD_DOCS=OFF
                     -DBUILD_EXAMPLES=OFF
                     -DBUILD_PERF_TESTS=OFF
                     -DBUILD_opencv_world=OFF
                     -DBUILD_opencv_apps=OFF
                     -DBUILD_opencv_python2=OFF
                     -DBUILD_opencv_python3=OFF
                     -DBUILD_opencv_gapi=OFF
                     -DBUILD_OPENEXR=OFF
                     -DWITH_GDAL=OFF
                     -DWITH_CUDA=OFF
                     -DWITH_IPP=OFF
                     -DWITH_OPENMP=OFF
                     -DWITH_GTK=OFF
                     -DWITH_GTK_2_X=OFF
                     -DWITH_OPENGL=OFF
                     -DWITH_GSTREAMER=OFF
                     -DWITH_OPENEXR=OFF
                     -DWITH_EIGEN=OFF
                     -DWITH_PROTOBUF=OFF
                     -DWITH_ADE=OFF
                     -DWITH_TIFF=OFF
                     -DWITH_JPEG=OFF
                     -DWITH_PNG=OFF
                     -DWITH_WEBP=OFF
                     -DWITH_OPENJPEG=OFF
                     -DWITH_JASPER=OFF
                     -DWITH_OPENEXR=OFF
                     -DWITH_JPEGXL=OFF

    LOG_DOWNLOAD          TRUE
    LOG_UPDATE            TRUE
    LOG_CONFIGURE         TRUE
    LOG_BUILD             TRUE
    LOG_INSTALL           TRUE
    LOG_MERGED_STDOUTERR  TRUE
    LOG_OUTPUT_ON_FAILURE TRUE

    DOWNLOAD_EXTRACT_TIMESTAMP ON
)

externalproject_add(
    dlib
    URL               ${CMAKE_SOURCE_DIR}/externals/dlib-19.13.tar.bz2
    PREFIX            ${CMAKE_BINARY_DIR}/dlib
    INSTALL_DIR       ${CMAKE_INSTALL_PREFIX}

    TEST_COMMAND      ""

    CMAKE_ARGS       -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
                     -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
                     -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                     -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                     -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
                     -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                     -DCMAKE_POLICY_VERSION_MINIMUM=3.5
                     -DBUILD_SHARED_LIBS=OFF
                     -DBUILD_TESTS=OFF
                     -DBUILD_DOCS=OFF
                     -DBUILD_EXAMPLES=OFF
                     -DBUILD_PERF_TESTS=OFF
                     -DDLIB_USE_CUDA=OFF

    LOG_DOWNLOAD          TRUE
    LOG_UPDATE            TRUE
    LOG_CONFIGURE         TRUE
    LOG_BUILD             TRUE
    LOG_INSTALL           TRUE
    LOG_MERGED_STDOUTERR  TRUE
    LOG_OUTPUT_ON_FAILURE TRUE

    DOWNLOAD_EXTRACT_TIMESTAMP ON

    DEPENDS openblas
)

externalproject_add(
    boost
    URL ${CMAKE_SOURCE_DIR}/externals/boost-1.88.0-cmake.tar.gz
    PREFIX ${CMAKE_BINARY_DIR}/${PROJECT_NAME} # Ensure unique prefix for each target
    INSTALL_DIR ${CMAKE_INSTALL_PREFIX}

    CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
    -DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=OFF
    -DBUILD_TESTING=OFF
    -DBOOST_ENABLE_PYTHON=OFF
    -DBOOST_ENABLE_MPI=OFF
    -DBOOST_INCLUDE_LIBRARIES=filesystem

    LOG_DOWNLOAD TRUE
    LOG_UPDATE TRUE
    LOG_CONFIGURE TRUE
    LOG_BUILD TRUE
    LOG_INSTALL TRUE
    LOG_MERGED_STDOUTERR TRUE
    LOG_OUTPUT_ON_FAILURE TRUE

    DOWNLOAD_EXTRACT_TIMESTAMP ON
)