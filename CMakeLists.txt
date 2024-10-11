cmake_minimum_required(VERSION 3.5) # CMake install : https://cmake.org/download/
project(QtYOLO LANGUAGES CXX)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_PREFIX_PATH "D:/Tools/QT/5.14.2/msvc2017_64") # Qt Kit Dir
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Qt5 COMPONENTS Widgets REQUIRED) # Qt COMPONENTS
aux_source_directory(./src srcs)

# Specify MSVC UTF-8 encoding   
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

# OpenCV
set(OpenCV_DIR "D:/Tools/opencv/opencv/build") 
find_package(OpenCV REQUIRED)

# onnxruntime
set(ONNXRUNTIME_DIR "D:/Work/YOLO/yolov8/v8Ort/Include")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/onnxruntime")
set(ONNXRUNTIME_LIB_DIR "D:/Work/YOLO/yolov8/v8Ort/Thirdlib/Release_x64")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")

list(APPEND ALL_INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

list(APPEND ALL_LIBS
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIB}
)


aux_source_directory(./include includes)

add_executable(${PROJECT_NAME}
    WIN32 # If you need a terminal for debug, please comment this statement 
    ${srcs} 
    ${includes}
    # ${CMAKE_CURRENT_SOURCE_DIR}/include/yolov8_onnx.h
    # ${CMAKE_CURRENT_SOURCE_DIR}/include/yolov8_utils.h
    # ${CMAKE_CURRENT_SOURCE_DIR}/include/yolov8_onnx.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/include/yolov8_utils.cpp
) 

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${ALL_INCLUDE_DIRS}
)

target_link_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${ONNXRUNTIME_LIB_DIR}
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    ${ALL_LIBS}
    Qt5::Widgets
)
# target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Widgets) # Qt5 Shared Library