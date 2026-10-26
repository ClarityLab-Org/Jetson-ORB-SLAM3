# Our own measurement tools (plan/paper_runbook.md, Part R3).
# A separate CMake project: it links against the already-built libORB_SLAM3.so of the repository
# and does not touch the repository's own CMakeLists.txt or sources.
#
#   cmake -S eval/benchmark/cpp -B eval/benchmark/cpp/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build eval/benchmark/cpp/build -j2
cmake_minimum_required(VERSION 3.16)
project(orbslam3_repro_tools CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "repository root (where lib/libORB_SLAM3.so is)")
set(ORB_LIB "${ORB_ROOT}/lib/libORB_SLAM3.so")
if(NOT EXISTS "${ORB_LIB}")
  message(FATAL_ERROR "${ORB_LIB} not found: build the repository first (runbook Stage 1, Part D)")
endif()

find_package(OpenCV 4 REQUIRED)
find_package(realsense2 QUIET)

set(ORB_RPATH "${ORB_ROOT}/lib;${ORB_ROOT}/Thirdparty/DBoW2/lib;${ORB_ROOT}/Thirdparty/g2o/lib;/usr/local/cuda/lib64;/usr/local/lib")

# feature_equivalence: uses only ORB_SLAM3::ORBextractor (include/ORBextractor.h needs OpenCV only)
add_executable(feature_equivalence feature_equivalence.cc)
target_include_directories(feature_equivalence PRIVATE "${ORB_ROOT}/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(feature_equivalence "${ORB_LIB}" ${OpenCV_LIBS})
# libORB_SLAM3.so's own dependencies (CUDA, TensorRT, Pangolin, g2o, DBoW2) are resolved at run
# time through its RUNPATH; we do not need their symbols at link time.
target_link_options(feature_equivalence PRIVATE "LINKER:--allow-shlib-undefined")
set_target_properties(feature_equivalence PROPERTIES BUILD_RPATH "${ORB_RPATH}")

if(realsense2_FOUND)
  add_executable(d435_record_euroc d435_record_euroc.cc)
  target_include_directories(d435_record_euroc PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(d435_record_euroc ${realsense2_LIBRARY} ${OpenCV_LIBS} pthread)
  set_target_properties(d435_record_euroc PROPERTIES BUILD_RPATH "/usr/local/lib")
else()
  message(WARNING "librealsense not found: d435_record_euroc is not built")
endif()
