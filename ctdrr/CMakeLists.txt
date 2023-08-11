cmake_minimum_required(VERSION 3.10.2)

project(ctdrr)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(ctdrr ctdrr.cxx )

target_link_libraries(ctdrr ${ITK_LIBRARIES})