cmake_minimum_required(VERSION 3.18)

project(SVIpcReceiver LANGUAGES CXX)

set(DIGIVIEW_COMMONS_HEADER
    "${CMAKE_CURRENT_SOURCE_DIR}/digiview_commons/ipc_wire_structs.hpp")
if(NOT EXISTS "${DIGIVIEW_COMMONS_HEADER}")
    message(FATAL_ERROR
        "The local digiview_commons submodule is not initialized. Run:\n"
        "  git submodule update --init --recursive")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(sv_ipc_receiver STATIC sv_ipc_receiver.cpp)
add_library(SVIpcReceiver::sv_ipc_receiver ALIAS sv_ipc_receiver)

target_compile_features(sv_ipc_receiver PUBLIC cxx_std_11)
target_include_directories(sv_ipc_receiver
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(sv_ipc_receiver
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver)

set_target_properties(sv_ipc_receiver PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(SV_IPC_RECEIVER_BUILD_EXAMPLE "Build the receive_frame example" OFF)
if(SV_IPC_RECEIVER_BUILD_EXAMPLE)
    add_executable(receive_frame_example examples/receive_frame.cpp)
    target_link_libraries(receive_frame_example
        PRIVATE
            SVIpcReceiver::sv_ipc_receiver)
endif()

option(SV_IPC_RECEIVER_BUILD_TEST_TOOLS "Build receiver test and C ABI bridge tools" OFF)
if(SV_IPC_RECEIVER_BUILD_TEST_TOOLS)
    find_package(ZLIB REQUIRED)

    add_executable(sv_ipc_receiver_test test.cpp)
    target_link_libraries(sv_ipc_receiver_test
        PRIVATE
            SVIpcReceiver::sv_ipc_receiver
            ZLIB::ZLIB)

    add_library(sv_ipc_receiver_test_bridge SHARED sv_ipc_receiver_test_bridge.cpp)
    target_link_libraries(sv_ipc_receiver_test_bridge
        PRIVATE
            SVIpcReceiver::sv_ipc_receiver)
endif()
