#include "sv_ipc_receiver.hpp"

#include <cstdarg>
#include <limits>
#include <memory>
#include <sys/stat.h>

bool SVGpuIpcReceiver::open_log_file() {
    if (log_file != nullptr) {
        return true;
    }

    expanded_log_path = log_path;
    if (!expanded_log_path.empty() && expanded_log_path.front() == '~') {
        const char *home = std::getenv("HOME");
        if (home == nullptr || home[0] == '\0') {
            std::fprintf(
                stderr,
                "SVGpuIpcReceiver[%s]: file logging disabled: HOME is not set for %s\n",
                socket_path.c_str(),
                log_path.c_str());
            return false;
        }
        if (expanded_log_path.size() == 1) {
            expanded_log_path = home;
        } else if (expanded_log_path[1] == '/') {
            expanded_log_path = std::string(home) + expanded_log_path.substr(1);
        } else {
            std::fprintf(
                stderr,
                "SVGpuIpcReceiver[%s]: file logging disabled: unsupported path %s\n",
                socket_path.c_str(),
                log_path.c_str());
            return false;
        }
    }

    log_file = std::fopen(expanded_log_path.c_str(), "a");
    if (log_file == nullptr) {
        const int saved_errno = errno;
        std::fprintf(
            stderr,
            "SVGpuIpcReceiver[%s]: file logging disabled: fopen(%s) failed: %s (%d)\n",
            socket_path.c_str(),
            expanded_log_path.c_str(),
            std::strerror(saved_errno),
            saved_errno);
        return false;
    }

    return true;
}

void SVGpuIpcReceiver::close_log_file() {
    if (log_file == nullptr) {
        return;
    }

    FILE *file = log_file;
    log_file = nullptr;
    std::fclose(file);
}

void SVGpuIpcReceiver::logf(const char *format, ...) const {
    FILE *const output = log_file != nullptr ? log_file : stderr;

    std::va_list args;
    va_start(args, format);
    std::fprintf(output, "SVGpuIpcReceiver[%s]: ", socket_path.c_str());
    std::vfprintf(output, format, args);
    std::fputc('\n', output);
    std::fflush(output);
    va_end(args);
}

// Helper: receive one fd via SCM_RIGHTS
int SVGpuIpcReceiver::recv_fd() {
    if (socket_fd < 0) {
        logf("recv_fd: socket is not connected");
        return -1;
    }

    char buf[CMSG_SPACE(sizeof(int))];
    char data[1] = {};
    struct iovec io;
    io.iov_base = data;
    io.iov_len = sizeof(data);
    struct msghdr msg;
    std::memset(buf, 0, sizeof(buf));
    std::memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);
    const ssize_t n = recvmsg(socket_fd, &msg, 0);
    if (n <= 0) {
        if (n == 0) {
            logf("recv_fd: EOF from sender while waiting for shared fd");
        } else {
            const int saved_errno = errno;
            logf("recv_fd: recvmsg failed: %s (%d)", std::strerror(saved_errno), saved_errno);
            errno = saved_errno;
        }
        return -1;
    }
    if ((msg.msg_flags & MSG_CTRUNC) != 0) {
        logf("recv_fd: control message truncated after %zd data bytes", n);
        return -1;
    }

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg) {
        logf("recv_fd: missing control message after %zd data bytes", n);
        return -1;
    }
    if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS ||
        cmsg->cmsg_len < CMSG_LEN(sizeof(int))) {
        logf(
            "recv_fd: invalid control message level=%d type=%d len=%zu expected_len=%zu",
            cmsg->cmsg_level,
            cmsg->cmsg_type,
            static_cast<size_t>(cmsg->cmsg_len),
            static_cast<size_t>(CMSG_LEN(sizeof(int))));
        return -1;
    }

    int fd = -1;
    std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
    return fd;
}

// Helper: receive exactly size bytes
bool SVGpuIpcReceiver::recv_metadata() {
    if (socket_fd < 0) {
        logf("recv_metadata: socket is not connected");
        return false;
    }

    char* ptr = reinterpret_cast<char*>(&metadata);
    size_t remaining = sizeof(metadata);
    while (remaining > 0) {
        const ssize_t n = recv(socket_fd, ptr, remaining, 0);
        if (n < 0) {
            const int saved_errno = errno;
            logf(
                "recv_metadata: recv failed with %zu bytes remaining: %s (%d)",
                remaining,
                std::strerror(saved_errno),
                saved_errno);
            errno = saved_errno;
            return false;
        }
        if (n == 0) {
            logf(
                "recv_metadata: EOF while reading metadata, %zu bytes still expected",
                remaining);
            return false;
        }
        ptr += n;
        remaining -= n;
    }
    return true;
}

bool SVGpuIpcReceiver::send_ack() {
    std::memset(&ack, 0, sizeof(ack));
    std::strncpy(ack.message, "ACK", sizeof(ack.message) - 1);
    ssize_t n = send(socket_fd, &ack, sizeof(ack), 0);
    if (n < 0) {
        const int saved_errno = errno;
        logf("send_ack: send failed: %s (%d)", std::strerror(saved_errno), saved_errno);
        return false;
    }
    if (n != static_cast<ssize_t>(sizeof(ack))) {
        logf("send_ack: partial send %zd of %zu bytes", n, sizeof(ack));
        return false;
    }
    return true;
}

bool SVGpuIpcReceiver::wait_for_sender() {
    open_log_file();

    if (socket_fd >= 0) {
        logf("wait_for_sender: already connected on socket fd=%d", socket_fd);
        return true;
    }

    logf("wait_for_sender: creating UNIX socket for %s", socket_path.c_str());
    socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        const int saved_errno = errno;
        logf("wait_for_sender: socket creation failed: %s (%d)", std::strerror(saved_errno), saved_errno);
        errno = saved_errno;
        return false;
    }

    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    if (socket_path.size() >= sizeof(addr.sun_path)) {
        logf(
            "wait_for_sender: socket path too long (%zu >= %zu): %s",
            socket_path.size(),
            sizeof(addr.sun_path),
            socket_path.c_str());
        cleanup();
        return false;
    }
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
    size_t retry = 0;
    logf("wait_for_sender: attempting to connect");
    while (connect(socket_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        const int saved_errno = errno;
        ++retry;
        logf(
            "wait_for_sender: connect attempt %zu failed: %s (%d)",
            retry,
            std::strerror(saved_errno),
            saved_errno);
        if (saved_errno == EINTR) {
            cleanup();
            errno = saved_errno;
            return false;
        }
        if (usleep(100000) != 0 && errno == EINTR) {
            const int interrupted_errno = errno;
            logf(
                "wait_for_sender: retry delay interrupted: %s (%d)",
                std::strerror(interrupted_errno),
                interrupted_errno);
            cleanup();
            errno = interrupted_errno;
            return false;
        }
    }

    logf("wait_for_sender: connected on socket fd=%d", socket_fd);

    // Initialize CUDA driver.
    CUresult cres;
    const char *errStr = nullptr;
    logf("wait_for_sender: initializing CUDA driver");
    cres = cuInit(0);
    if (cres != CUDA_SUCCESS) {
        cuGetErrorString(cres, &errStr);
        logf("wait_for_sender: cuInit failed: %d (%s)", cres, errStr ? errStr : "unknown");
        cleanup();
        return false;
    }

    CUdevice cuDevice;
    cres = cuDeviceGet(&cuDevice, 0);
    if (cres != CUDA_SUCCESS) {
        cuGetErrorString(cres, &errStr);
        logf("wait_for_sender: cuDeviceGet failed: %d (%s)", cres, errStr ? errStr : "unknown");
        cleanup();
        return false;
    }

    logf("wait_for_sender: CUDA device 0 ready");
    return true;
}

// Main frame receive logic: fills digiview_frame
bool SVGpuIpcReceiver::receive_frame(digiview_frame &frame) {
    if (socket_fd < 0) {
        logf("receive_frame: socket is not connected");
        return false;
    }

    int share_fd = recv_fd();
    if (share_fd < 0) {
        return false;
    }

    if (!recv_metadata()) {
        if (close(share_fd) != 0) {
            const int saved_errno = errno;
            logf(
                "receive_frame: close(%d) after metadata failure failed: %s (%d)",
                share_fd,
                std::strerror(saved_errno),
                saved_errno);
        }
        return false;
    }

    if (metadata.start_byte != 0xFF) {
        logf("receive_frame: rejecting invalid metadata start byte 0x%02X", metadata.start_byte);
        if (close(share_fd) != 0) {
            const int saved_errno = errno;
            logf(
                "receive_frame: close(%d) after invalid metadata failed: %s (%d)",
                share_fd,
                std::strerror(saved_errno),
                saved_errno);
        }
        return false;
    }

    const auto pixel_format = ipc::gpu_frame_pixel_format_from_flags(metadata.flags);
    if (pixel_format != ipc::GpuFramePixelFormat::kBgr8 &&
        pixel_format != ipc::GpuFramePixelFormat::kBgra8) {
        logf(
            "receive_frame: rejecting unsupported GPU frame format flags=0x%08X%s",
            static_cast<unsigned int>(metadata.flags),
            metadata.flags == 0 ? " (legacy or unspecified)" : "");
        if (close(share_fd) != 0) {
            const int saved_errno = errno;
            logf(
                "receive_frame: close(%d) after unsupported format failed: %s (%d)",
                share_fd,
                std::strerror(saved_errno),
                saved_errno);
        }
        return false;
    }

    const size_t pixel_size = ipc::gpu_frame_bytes_per_pixel(pixel_format);
    const int width = metadata.frame_width;
    const int height = metadata.frame_height;
    if (width <= 0 || height <= 0) {
        logf("receive_frame: invalid frame dimensions width=%d height=%d", width, height);
        if (close(share_fd) != 0) {
            const int saved_errno = errno;
            logf(
                "receive_frame: close(%d) after invalid metadata failed: %s (%d)",
                share_fd,
                std::strerror(saved_errno),
                saved_errno);
        }
        return false;
    }

    const size_t width_size = static_cast<size_t>(width);
    const size_t height_size = static_cast<size_t>(height);
    if (width_size > std::numeric_limits<size_t>::max() / height_size / pixel_size) {
        logf(
            "receive_frame: frame size overflow for width=%d height=%d pixel_size=%zu",
            width,
            height,
            pixel_size);
        if (close(share_fd) != 0) {
            const int saved_errno = errno;
            logf(
                "receive_frame: close(%d) after size overflow failed: %s (%d)",
                share_fd,
                std::strerror(saved_errno),
                saved_errno);
        }
        return false;
    }

    if (width_size > static_cast<size_t>(std::numeric_limits<int32_t>::max()) / pixel_size) {
        logf("receive_frame: pitch overflow for width=%d pixel_size=%zu", width, pixel_size);
        if (close(share_fd) != 0) {
            const int saved_errno = errno;
            logf(
                "receive_frame: close(%d) after pitch overflow failed: %s (%d)",
                share_fd,
                std::strerror(saved_errno),
                saved_errno);
        }
        return false;
    }

    const size_t frame_bytes = width_size * height_size * pixel_size;
    CUmemGenericAllocationHandle handle = 0;
    bool handle_imported = false;
    CUresult cres;
    const char* errStr = nullptr;
    CUdeviceptr devPtr = 0;
    bool address_reserved = false;
    bool memory_mapped = false;
    std::unique_ptr<unsigned char, decltype(&std::free)> host_buffer(nullptr, &std::free);
    size_t granularity = 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    size_t allocSize = 0;
    bool success = false;

    do {
        void* osHandle = reinterpret_cast<void*>(static_cast<intptr_t>(share_fd));
        cres = cuMemImportFromShareableHandle(&handle, osHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
        if (close(share_fd) != 0) {
            const int saved_errno = errno;
            logf(
                "receive_frame: close(%d) after import attempt failed: %s (%d)",
                share_fd,
                std::strerror(saved_errno),
                saved_errno);
        }
        if (cres != CUDA_SUCCESS) {
            cuGetErrorString(cres, &errStr);
            logf("receive_frame: cuMemImportFromShareableHandle failed: %d (%s)", cres, errStr ? errStr : "unknown");
            break;
        }
        handle_imported = true;

        cres = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (cres != CUDA_SUCCESS) {
            cuGetErrorString(cres, &errStr);
            logf("receive_frame: cuMemGetAllocationGranularity failed: %d (%s)", cres, errStr ? errStr : "unknown");
            break;
        }

        if (granularity == 0 ||
            frame_bytes > std::numeric_limits<size_t>::max() - (granularity - 1)) {
            logf(
                "receive_frame: allocation size overflow for frame_bytes=%zu granularity=%zu",
                frame_bytes,
                granularity);
            break;
        }
        allocSize = ((frame_bytes + granularity - 1) / granularity) * granularity;

        cres = cuMemAddressReserve(&devPtr, allocSize, granularity, 0, 0);
        if (cres != CUDA_SUCCESS) {
            cuGetErrorString(cres, &errStr);
            logf("receive_frame: cuMemAddressReserve failed: %d (%s)", cres, errStr ? errStr : "unknown");
            break;
        }
        address_reserved = true;

        cres = cuMemMap(devPtr, allocSize, 0, handle, 0);
        if (cres != CUDA_SUCCESS) {
            cuGetErrorString(cres, &errStr);
            logf("receive_frame: cuMemMap failed: %d (%s)", cres, errStr ? errStr : "unknown");
            break;
        }
        memory_mapped = true;

        CUmemAccessDesc accessDesc{};
        accessDesc.location = prop.location;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cres = cuMemSetAccess(devPtr, allocSize, &accessDesc, 1);
        if (cres != CUDA_SUCCESS) {
            cuGetErrorString(cres, &errStr);
            logf("receive_frame: cuMemSetAccess failed: %d (%s)", cres, errStr ? errStr : "unknown");
            break;
        }

        host_buffer.reset(static_cast<unsigned char*>(std::malloc(frame_bytes)));
        if (!host_buffer) {
            logf("receive_frame: malloc(%zu) failed", frame_bytes);
            break;
        }

        cudaError_t ce = cudaMemcpy(host_buffer.get(), reinterpret_cast<void*>(devPtr), frame_bytes, cudaMemcpyDeviceToHost);
        if (ce != cudaSuccess) {
            logf("receive_frame: cudaMemcpy failed: %s", cudaGetErrorString(ce));
            break;
        }
        if (!send_ack()) {
            logf("receive_frame: failed to send ACK");
            break;
        }

        frame.timestamp = metadata.timestamp;
        frame.system_coordinate[0] = metadata.system_coordinate[0];
        frame.system_coordinate[1] = metadata.system_coordinate[1];
        frame.system_altitude = metadata.system_altitude;
        frame.home_altitude = metadata.home_altitude;
        for (int i = 0; i < 3; ++i) {
            frame.acc[i] = metadata.acc[i];
            frame.vel[i] = metadata.vel[i];
            frame.dir[i] = metadata.dir[i];
            frame.auto_pilot_euler[i] = metadata.auto_pilot_euler[i];
            frame.auto_pilot_acc[i] = metadata.auto_pilot_acc[i];
        }
        frame.data = reinterpret_cast<Npp8u*>(host_buffer.release());
        frame.width = width;
        frame.height = height;
        frame.pixel_format = static_cast<int32_t>(pixel_format);
        frame.pitch = static_cast<int32_t>(width_size * pixel_size);

        success = true;
    } while (false);

    if (memory_mapped) {
        cres = cuMemUnmap(devPtr, allocSize);
        if (cres != CUDA_SUCCESS) {
            cuGetErrorString(cres, &errStr);
            logf("receive_frame: cuMemUnmap failed: %d (%s)", cres, errStr ? errStr : "unknown");
        }
    }

    if (address_reserved) {
        cres = cuMemAddressFree(devPtr, allocSize);
        if (cres != CUDA_SUCCESS) {
            cuGetErrorString(cres, &errStr);
            logf("receive_frame: cuMemAddressFree failed: %d (%s)", cres, errStr ? errStr : "unknown");
        }
    }

    if (handle_imported) {
        cres = cuMemRelease(handle);
        if (cres != CUDA_SUCCESS) {
            cuGetErrorString(cres, &errStr);
            logf("receive_frame: cuMemRelease failed: %d (%s)", cres, errStr ? errStr : "unknown");
        }
    }

    if (!success) {
        return false;
    }
    return true;
}

void SVGpuIpcReceiver::cleanup() {
    if (socket_fd >= 0) {
        const int fd = socket_fd;
        socket_fd = -1;
        if (close(fd) != 0) {
            const int saved_errno = errno;
            logf("cleanup: close(%d) failed: %s (%d)", fd, std::strerror(saved_errno), saved_errno);
        } else {
            logf("cleanup: closed socket fd=%d", fd);
        }
    }

    close_log_file();
}
