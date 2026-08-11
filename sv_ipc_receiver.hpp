#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>

#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <cuda.h>
#include <npp.h>

#include <fcntl.h>
#include <iostream>

#include <digiview_commons/ipc_wire_structs.hpp>

struct digiview_frame {
    uint64_t timestamp = 0;
    float    acc[3] = {0, 0, 0};
    float    vel[3] = {0, 0, 0};
    float    dir[3] = {0, 0, 0};
    float    system_coordinate[2] = {0, 0}; 
    float    system_altitude = 0; 
    float    home_altitude = 0; 
    float    auto_pilot_euler[3] = {0, 0, 0}; 
    float    auto_pilot_acc[3] = {0, 0, 0};
    Npp8u   *data = nullptr;
    int32_t  width = 0;
    int32_t  height = 0;
    int32_t  pixel_format = -1; // ipc::GpuFramePixelFormat underlying value
    int32_t  pitch = 0;
};


class SVGpuIpcReceiver {
public:
    SVGpuIpcReceiver(std::string socket_path = "/tmp/source_camera_0_socket0")
        : socket_path(socket_path) {}
    ~SVGpuIpcReceiver() { cleanup(); }

    bool wait_for_sender();
    // On success, the caller owns frame.data and must release it with std::free.
    bool receive_frame(digiview_frame &frame);
    void cleanup();

private:
    void logf(const char *format, ...) const;
    bool open_log_file();
    void close_log_file();
    int recv_fd();
    bool recv_metadata();
    bool send_ack();

    digiview_metadata metadata{};
    acknowledgment ack{};

    std::string socket_path;
    std::string log_path = "~/svipc.log";
    std::string expanded_log_path;
    FILE *log_file = nullptr;
    int socket_fd = -1;
};
