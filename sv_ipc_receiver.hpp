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

#include <nvbufsurface.h>
#include <cuda_runtime.h>
#include <npp.h>

static constexpr int32_t CUDA_MALLOC_DEVICE  = 0;
static constexpr int32_t CUDA_MALLOC_HOST    = 1;
static constexpr int32_t CUDA_MALLOC_MANAGED = 2;

struct digiview_frame {
    uint64_t timestamp = 0;
    float    acc[3] = {0, 0, 0}; // the frame's acceleration vector in the global frame
    float    vel[3] = {0, 0, 0}; // the frame's velocity vector in the global frame
    float    dir[3] = {0, 0, 0}; // the frame's direction vector in the global frame
    Npp8u   *data = nullptr;
    int32_t  width = 0;
    int32_t  height = 0;
    int32_t  pixel_format = -1;
    int32_t  pitch = 0;
};

class SVIpcReceiver {
    public:
        SVIpcReceiver(int32_t cuda_alloc_type = CUDA_MALLOC_DEVICE, std::string socket_path = "/tmp/source_camera_0_socket")
            : cuda_alloc_type(cuda_alloc_type), socket_path(socket_path) {}
        ~SVIpcReceiver() {}

        bool wait_for_sender();
        bool receive_frame(digiview_frame &frame);
        void cleanup();

    private:
        int recv_fd();
        bool recv_metadata();
        bool send_ack();
        bool allocate_frame(int32_t height, int32_t pitch);

        // Make sure this matches the sender-side wire header layout exactly
        struct digiview_metadata {
            uint8_t               start_byte = 0xFF;
            uint64_t              timestamp;
            float                 acc[3];
            float                 vel[3];
            float                 dir[3];
            NvBufSurfaceMapParams params;
            int32_t               flags; // e.g. last frame
        } metadata;

        struct acknowledgment {
            char message[16];
        } ack;

        std::string socket_path;
        int socket_fd = -1;
        NvBufSurface *nvbuf_surf = nullptr;
        Npp8u *copied_frame = nullptr;
        int32_t cuda_alloc_type = 0;
};