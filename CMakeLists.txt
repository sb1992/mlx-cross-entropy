cmake_minimum_required(VERSION 3.27)
project(_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(BUILD_SHARED_LIBS "Build as shared library" ON)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

add_library(mlx_cce_ext)
target_sources(mlx_cce_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR}/csrc/cce/cce.cpp)
target_include_directories(mlx_cce_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR}/csrc)
target_link_libraries(mlx_cce_ext PUBLIC mlx)

if(MLX_BUILD_METAL)
  set(CCE_METAL_INCDIRS ${PROJECT_SOURCE_DIR}/csrc ${MLX_INCLUDE_DIRS})
  set(METALLIB_OUTPUT "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/mlx_cce_ext.metallib")
  add_custom_command(
    OUTPUT ${METALLIB_OUTPUT}
    COMMAND xcrun -sdk macosx metal
      "$<LIST:TRANSFORM,${CCE_METAL_INCDIRS},PREPEND,-I>"
      -O2 -Wall -Wextra -Wno-c++17-extensions
      ${CMAKE_CURRENT_LIST_DIR}/csrc/cce/cce.metal
      -o ${METALLIB_OUTPUT}
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/csrc/cce/cce.metal
    COMMAND_EXPAND_LISTS
    COMMENT "Building mlx_cce_ext.metallib"
    VERBATIM)
  add_custom_target(mlx_cce_ext_metallib DEPENDS ${METALLIB_OUTPUT})
  add_dependencies(mlx_cce_ext mlx_cce_ext_metallib)
endif()

nanobind_add_module(
  _ext NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/csrc/bindings.cpp)
target_link_libraries(_ext PRIVATE mlx_cce_ext)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
