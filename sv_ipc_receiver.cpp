#include "sv_ipc_receiver.hpp"

bool SVIpcReceiver::wait_for_sender() {
    if (socket_fd >= 0) {
        return true;
    }

    socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        perror("SVIpcReceiver::wait_for_sender: socket");
        return false;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    printf("SVIpcReceiver: Connecting to sender at %s...\n", socket_path.c_str());

    int retry = 0;
    int ret;
    do {
        usleep(200000); // 200 ms
        ret = connect(socket_fd,
                      reinterpret_cast<struct sockaddr*>(&addr),
                      sizeof(addr));
        if (ret < 0) retry++;
    } while (ret < 0 && retry < 50);

    if (ret < 0) {
        printf("SVIpcReceiver::wait_for_sender: connect failed: %s\n",
               strerror(errno));
        close(socket_fd);
        socket_fd = -1;
        return false;
    }

    printf("SVIpcReceiver: Connected to sender.\n");
    return true;
}

bool SVIpcReceiver::receive_frame(digiview_frame &frame) {
    // Make sure we are connected
    if (socket_fd < 0) {
        printf("SVIpcReceiver::receive_frame: not connected\n");
        return false;
    }

    // Receive dmabuf fd
    int dmabuf_fd = recv_fd();
    if (dmabuf_fd < 0) {
        printf("SVIpcReceiver: No more frames or error, exiting.\n");
        return false;
    }

    // Read metadata header
    if (!recv_metadata()) {
        printf("SVIpcReceiver::receive_frame: Failed to receive frame header\n");
        close(dmabuf_fd);
        return false;
    }

    // Import NvBufSurface from received fd and parameters
    NvBufSurface *surf = nullptr;
    metadata.params.fd = dmabuf_fd;

    if (NvBufSurfaceImport(&surf, &metadata.params) != 0) {
        printf("SVIpcReceiver::receive_frame: NvBufSurfaceImport failed\n");
        close(dmabuf_fd);
        return false;
    }

    // Mark buffer as containing 1 valid frame and map it for CPU access
    surf->numFilled = 1;
    if (NvBufSurfaceMap(surf, 0, 0, NVBUF_MAP_READ) != 0) {
        printf("SVIpcReceiver::receive_frame: NvBufSurfaceMap failed\n");
        NvBufSurfaceDestroy(surf);
        return false;
    }

    // Synchronize the buffer for CPU access and take pointer to data
    NvBufSurfaceSyncForCpu(surf, 0, 0);
    void* ptr = surf->surfaceList[0].mappedAddr.addr[0];
    if (!ptr) {
        printf("SVIpcReceiver::receive_frame: mappedAddr.addr[0] is null\n");
        NvBufSurfaceUnMap(surf, 0, 0);
        NvBufSurfaceDestroy(surf);
        return false;
    }

    uint8_t* src      = static_cast<uint8_t*>(ptr);
    uint32_t width    = surf->surfaceList[0].width;
    uint32_t height   = surf->surfaceList[0].height;
    uint32_t pitch    = surf->surfaceList[0].pitch;

    size_t buffer_size = static_cast<size_t>(pitch) * height;

    printf("SVIpcReceiver: Received frame: timestamp=%lu, size=%dx%d, pitch=%d, pixel_format=%d\n",
           metadata.timestamp, width, height, pitch, metadata.params.colorFormat);

    // Allocate frame buffer if not already done
    if (!copied_frame) {
        if (!allocate_frame(height, pitch)) {
            printf("SVIpcReceiver::receive_frame: allocate_frame failed\n");
            NvBufSurfaceUnMap(surf, 0, 0);
            NvBufSurfaceDestroy(surf);
            return false;
        }
    }

    // Copy data to allocated frame buffer
    cudaError_t e;
    switch (cuda_alloc_type) {
        case CUDA_MALLOC_HOST:
            e = cudaMemcpy2D(copied_frame, pitch, src, pitch, pitch, height,
                cudaMemcpyHostToHost);
            break;
        case CUDA_MALLOC_DEVICE:
            e = cudaMemcpy2D(copied_frame, pitch, src, pitch, pitch, height,
                cudaMemcpyHostToDevice);
            break;
        case CUDA_MALLOC_MANAGED:
            e = cudaMemcpy2D(copied_frame, pitch, src, pitch, pitch, height,
                cudaMemcpyHostToHost);
            break;
        default:
            memcpy(copied_frame, src, buffer_size);
            e = cudaSuccess;
            break;
    }
    if (e != cudaSuccess) {
        printf("SVIpcReceiver::receive_frame: copy failed: %s\n", cudaGetErrorString(e));
        NvBufSurfaceUnMap(surf, 0, 0);
        NvBufSurfaceDestroy(surf);
        return false;
    }

    // Unmap and destroy NvBufSurface, close dmabuf fd
    NvBufSurfaceUnMap(surf, 0, 0);
    NvBufSurfaceDestroy(surf);
    surf = nullptr;
    close(dmabuf_fd);

    // Fill out digiview_frame structure
    frame.timestamp = metadata.timestamp;
    for (int i = 0; i < 3; ++i) {
        frame.acc[i] = metadata.acc[i];
        frame.vel[i] = metadata.vel[i];
        frame.dir[i] = metadata.dir[i];
    }
    frame.data = copied_frame;
    frame.width  = static_cast<int32_t>(width);
    frame.height = static_cast<int32_t>(height);
    frame.pixel_format = metadata.params.colorFormat;
    frame.pitch = static_cast<int32_t>(pitch);

    // 6) Send ACK back to sender
    if (!send_ack()) {
        printf("SVIpcReceiver::receive_frame: Failed to send ACK\n");
        return false;
    }

    return true;
}

void SVIpcReceiver::cleanup() {
    if (socket_fd >= 0) {
        close(socket_fd);
        socket_fd = -1;
    }
    if (nvbuf_surf) {
        NvBufSurfaceDestroy(nvbuf_surf);
        nvbuf_surf = nullptr;
    }
    if (copied_frame) {
        switch (cuda_alloc_type) {
            case CUDA_MALLOC_HOST:
                cudaFreeHost(copied_frame);
                break;
            case CUDA_MALLOC_DEVICE:
            case CUDA_MALLOC_MANAGED:
                cudaFree(copied_frame);
                break;
            default:
                free(copied_frame);
                break;
        }
        copied_frame = nullptr;
    }
}

int SVIpcReceiver::recv_fd() {
    char buf[CMSG_SPACE(sizeof(int))];
    char data[1]; // dummy payload

    struct iovec io;
    io.iov_base = data;
    io.iov_len  = sizeof(data);

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov    = &io;
    msg.msg_iovlen = 1;
    msg.msg_control    = buf;
    msg.msg_controllen = sizeof(buf);

    ssize_t n = recvmsg(socket_fd, &msg, 0);
    if (n <= 0) {
        if (n == 0) {
            printf("SVIpcReceiver::recv_fd: EOF from sender\n");
        } else {
            printf("SVIpcReceiver::recv_fd: recvmsg failed\n");
        }
        return -1;
    }

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg ||
        cmsg->cmsg_level != SOL_SOCKET ||
        cmsg->cmsg_type  != SCM_RIGHTS) {

        printf("recvFd: invalid control message\n");
        return -1;
    }

    int fd = -1;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
    return fd;
}

bool SVIpcReceiver::recv_metadata() {
    char *ptr = reinterpret_cast<char*>(&metadata);
    size_t remaining = sizeof(metadata);

    while (remaining > 0) {
        ssize_t n = recv(socket_fd, ptr, remaining, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            printf("SVIpcReceiver::recv_metadata: recv failed\n");
            return false;
        }
        if (n == 0) {
            printf("SVIpcReceiver::recv_metadata: EOF while reading\n");
            return false;
        }
        ptr       += n;
        remaining -= n;
    }
    return true;
}

bool SVIpcReceiver::send_ack() {
    strncpy(ack.message, "ACK", sizeof(ack.message));
    ssize_t n = send(socket_fd, &ack, sizeof(ack), 0);
    if (n != sizeof(ack)) {
        printf("SVIpcReceiver::send_ack: send failed\n");
        return false;
    }
    return true;
}

bool SVIpcReceiver::allocate_frame(int32_t height, int32_t pitch) {
    cudaError_t e;
    switch (cuda_alloc_type) {
        case CUDA_MALLOC_DEVICE:
            e = cudaMalloc((void **)&copied_frame, pitch * height);
            if (e != cudaSuccess) {
                printf("SVIpcReceiver::allocate_frame: cudaMalloc failed: %s\n", cudaGetErrorString(e));
                return false;
            }
            break;
        case CUDA_MALLOC_HOST:
            e = cudaMallocHost((void **)&copied_frame, pitch * height, cudaHostAllocDefault);
            if (e != cudaSuccess) {
                printf("SVIpcReceiver::allocate_frame: cudaMallocHost failed: %s\n", cudaGetErrorString(e));
                return false;
            }
            break;
        case CUDA_MALLOC_MANAGED:
            e = cudaHostAlloc((void **)&copied_frame, pitch * height, cudaHostAllocMapped);
            if (e != cudaSuccess) {
                printf("SVIpcReceiver::allocate_frame: cudaHostAlloc failed: %s\n", cudaGetErrorString(e));
                return false;
            }
            break;
        default:
            copied_frame = static_cast<Npp8u*>(malloc(pitch * height));
            if (!copied_frame) {
                printf("SVIpcReceiver::allocate_frame: malloc failed\n");
                return false;
            }
            break;
    }
    return true;
}