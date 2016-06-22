cmake_minimum_required(VERSION 2.6)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# OpenCV Config
find_package(OpenCV REQUIRED)
message("-- OPENCV include:   " ${OpenCV_INCLUDE_DIRS})
message("-- OPENCV libs dir:  " ${OpenCV_LIB_DIR})
message("-- OPENCV libs:   " ${OpenCV_LIBS} )

# Conditional clause and corresponding automatic includings
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIB_DIR})


ADD_DEFINITIONS(-DDEBUG -ggdb -O3 -Wall -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -ffast-math -fbranch-probabilities)

set(SRCS
    lib/ldb/Ldb.cpp
    lib/d_ldb/D_Ldb.cpp
    src/OpenABLE.cpp
)

ADD_EXECUTABLE(
    Test_OpenABLE
    Test_OpenABLE.cpp 
    ${SRCS}
)

TARGET_LINK_LIBRARIES(
    Test_OpenABLE
    ${OpenCV_LIBS}
)

################################

foreach(dir ${DIRS})
    FILE(GLOB files ${dir}/*.h*)
    LIST(APPEND h_files ${files})
endforeach(dir)
add_custom_target(TAGS etags --members -l c++ ${h_files} ${SRCS})
