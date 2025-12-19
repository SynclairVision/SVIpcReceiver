#pragma once
#include <lodepng.h>
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

struct digiview_frame {
    uint64_t timestamp = 0;
    float    acc[3] = {0, 0, 0};
    float    vel[3] = {0, 0, 0};
    float    dir[3] = {0, 0, 0};
    float    system_coordinate[2]; 
    float    system_altitude; 
    float    home_altitude;
    float    auto_pilot_euler[3]; 
    float    auto_pilot_acc[3];
    Npp8u   *data = nullptr;
    int32_t  width = 0;
    int32_t  height = 0;
    int32_t  pixel_format = -1;
    int32_t  pitch = 0;
};


class SVGpuIpcReceiver {
public:
    SVGpuIpcReceiver(std::string socket_path = "/tmp/source_camera_0_socket")
        : socket_path(socket_path) {}
    ~SVGpuIpcReceiver() {}

    bool wait_for_sender();
    bool receive_frame(digiview_frame &frame);
    void cleanup();

private:
    int recv_fd();
    bool recv_metadata();
    bool send_ack();

    struct digiview_metadata {
        uint8_t  start_byte = 0xFF;
        uint64_t timestamp;
        float    acc[3];
        float    vel[3];
        float    dir[3];
        float    system_coordinate[2]; 
        float    system_altitude; 
        float    home_altitude; 
        float    auto_pilot_euler[3]; 
        float    auto_pilot_acc[3]; 
        int32_t  frame_width;
        int32_t  frame_height;
        int32_t  flags;
    }metadata;

    struct acknowledgment {
        char message[16] = {0};
    }ack;

    std::string socket_path;
    int socket_fd = -1;
};
