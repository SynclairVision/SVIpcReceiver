#include "sv_ipc_receiver.hpp"

#include <sys/stat.h>

// Helper: receive one fd via SCM_RIGHTS
int SVGpuIpcReceiver::recv_fd() {
    char buf[CMSG_SPACE(sizeof(int))];
    char data[1];
    struct iovec io;
    io.iov_base = data;
    io.iov_len = sizeof(data);
    struct msghdr msg;
    std::memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);
    ssize_t n = recvmsg(socket_fd, &msg, 0);
    if (n <= 0) {
        if (n == 0) {
            printf("SVGpuIpcReceiver::recv_fd: EOF from sender\n");
        } else {
            perror("SVGpuIpcReceiver::recv_fd: recvmsg failed");
        }
        return -1;
    }
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
        fprintf(stderr, "SVGpuIpcReceiver::recv_fd: invalid control message\n");
        return -1;
    }
    int fd = -1;
    std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
    return fd;
}

// Helper: receive exactly size bytes
bool SVGpuIpcReceiver::recv_metadata() {
    char* ptr = reinterpret_cast<char*>(&metadata);
    size_t remaining = sizeof(metadata);
    while (remaining > 0) {
        ssize_t n = recv(socket_fd, ptr, remaining, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            perror("SVGpuIpcReceiver::recv_metadata: recv failed");
            return false;
        }
        if (n == 0) {
            fprintf(stderr, "SVGpuIpcReceiver::recv_metadata: EOF while reading\n");
            return false;
        }
        ptr += n;
        remaining -= n;
    }
    return true;
}

bool SVGpuIpcReceiver::send_ack() {
    strncpy(ack.message, "ACK", sizeof(ack.message));
    ssize_t n = send(socket_fd, &ack, sizeof(ack), 0);
    if (n != (ssize_t)sizeof(ack)) {
        perror("SVGpuIpcReceiver::send_ack: send failed");
        return false;
    }
    return true;
}

bool SVGpuIpcReceiver::wait_for_sender() {
    if (socket_fd >= 0) return true;
    socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        perror("SVGpuIpcReceiver::wait_for_sender: socket");
        return false;
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
    int retry = 0, ret;
    do {
        ret = connect(socket_fd, (struct sockaddr*)&addr, sizeof(addr));
        if (ret < 0) { usleep(100000); retry++; }
    } while (ret < 0 && retry < 50);
    if (ret < 0) {
        perror("SVGpuIpcReceiver::wait_for_sender: connect");
        close(socket_fd);
        socket_fd = -1;
        return false;
    }

    // Initialize CUDA driver & create context (driver API)
    CUresult cres;
    const char *errStr = nullptr;
    cres = cuInit(0);
    if (cres != CUDA_SUCCESS) {
        cuGetErrorString(cres, &errStr);
        std::fprintf(stderr, "cuInit failed: %d (%s)\n", cres, errStr ? errStr : "unknown");
        return false;
    }
    CUdevice cuDevice;
    cres = cuDeviceGet(&cuDevice, 0);
    if (cres != CUDA_SUCCESS) {
        cuGetErrorString(cres, &errStr);
        std::fprintf(stderr, "cuDeviceGet failed: %d (%s)\n", cres, errStr ? errStr : "unknown");
        return false;
    }
    printf("SVGpuIpcReceiver: Connected to sender and created CUDA context.\n");
    return true;
}

// Main frame receive logic: fills digiview_frame
bool SVGpuIpcReceiver::receive_frame(digiview_frame &frame) {
    int share_fd = recv_fd();
    if (share_fd < 0) {
        return false;
    }
    printf("SVGpuIpcReceiver: Received FD %d\n", share_fd);

    if (!recv_metadata()) {
        close(share_fd);
        return false;
    }
    printf("SVGpuIpcReceiver: Received metadata: ts=%llu, size=%dx%d, flags=%d\n",
           (unsigned long long)metadata.timestamp, metadata.frame_width, metadata.frame_height, metadata.flags);

    CUmemGenericAllocationHandle handle = 0;
    CUresult cres;
    const char* errStr = nullptr;
    void* osHandle = reinterpret_cast<void*>(static_cast<intptr_t>(share_fd));
    cres = cuMemImportFromShareableHandle(&handle, osHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    if (cres != CUDA_SUCCESS) {
        cuGetErrorString(cres, &errStr);
        std::fprintf(stderr, "cuMemImportFromShareableHandle failed: %d (%s)\n", cres, errStr ? errStr : "unknown");
        close(share_fd);
        return false;
    }
    int width = metadata.frame_width;
    int height = metadata.frame_height;
    size_t pixel_size = 3; 
    size_t frame_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * pixel_size;

    size_t granularity = 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    cres = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (cres != CUDA_SUCCESS) {
        cuGetErrorString(cres, &errStr);
        std::fprintf(stderr, "cuMemGetAllocationGranularity failed: %d (%s)\n", cres, errStr ? errStr : "unknown");
        return false;
    }
    size_t allocSize = ((frame_bytes + granularity - 1) / granularity) * granularity;

    CUdeviceptr devPtr = 0;
    cres = cuMemAddressReserve(&devPtr, allocSize, granularity, 0, 0);
    if (cres != CUDA_SUCCESS) {
        cuGetErrorString(cres, &errStr);
        std::fprintf(stderr, "cuMemAddressReserve failed: %d (%s)\n", cres, errStr ? errStr : "unknown");
        return false;
    }
    cres = cuMemMap(devPtr, allocSize, 0, handle, 0);
    if (cres != CUDA_SUCCESS) {
        cuGetErrorString(cres, &errStr);
        std::fprintf(stderr, "cuMemMap failed: %d (%s)\n", cres, errStr ? errStr : "unknown");
        return false;
    }
    CUmemAccessDesc accessDesc{};
    accessDesc.location = prop.location;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cres = cuMemSetAccess(devPtr, allocSize, &accessDesc, 1);
    if (cres != CUDA_SUCCESS) {
        cuGetErrorString(cres, &errStr);
        std::fprintf(stderr, "cuMemSetAccess failed: %d (%s)\n", cres, errStr ? errStr : "unknown");
        cuMemUnmap(devPtr, allocSize);
        return false;
    }

    unsigned char* host_buffer = (unsigned char*)malloc(frame_bytes);
    if (!host_buffer) {
        std::fprintf(stderr, "SVGpuIpcReceiver::receive_frame: malloc(%zu) failed\n", frame_bytes);
        cuMemUnmap(devPtr, allocSize);
        return false;
    }

    cudaError_t ce = cudaMemcpy(host_buffer, (void*)devPtr, frame_bytes, cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) {
        std::fprintf(stderr, "SVGpuIpcReceiver::receive_frame: cudaMemcpy failed: %s\n", cudaGetErrorString(ce));
        free(host_buffer);
        cuMemUnmap(devPtr, allocSize);
        return false;
    }
    cuMemUnmap(devPtr, allocSize);

    frame.timestamp = metadata.timestamp;
    for (int i = 0; i < 3; ++i) {
        frame.acc[i] = metadata.acc[i];
        frame.vel[i] = metadata.vel[i];
        frame.dir[i] = metadata.dir[i];
    }
    frame.data = reinterpret_cast<Npp8u*>(host_buffer);
    frame.width = width;
    frame.height = height;
    frame.pixel_format = 0; // assume RGB
    frame.pitch = static_cast<int32_t>(width * pixel_size);

    if (!send_ack()) {
        std::fprintf(stderr, "SVGpuIpcReceiver::receive_frame: Failed to send ACK\n");
        free(host_buffer);
        return false;
    }
    printf("SVGpuIpcReceiver: Sent ACK for ts=%llu\n", (unsigned long long)metadata.timestamp);

    return true;
}

void SVGpuIpcReceiver::cleanup() {
    if (socket_fd >= 0) close(socket_fd);
    socket_fd = -1;
}
