#include "sv_ipc_receiver_test_bridge.h"

#include "sv_ipc_receiver.hpp"

#include <cstdlib>
#include <new>

struct sv_ipc_receiver {
    explicit sv_ipc_receiver(const char *socket_path)
        : receiver(socket_path) {}

    SVGpuIpcReceiver receiver;
};

extern "C" sv_ipc_receiver_status sv_ipc_receiver_create(
    const char *socket_path,
    sv_ipc_receiver **receiver) {
    if (socket_path == nullptr || receiver == nullptr) {
        return SV_IPC_RECEIVER_STATUS_INVALID_ARGUMENT;
    }

    *receiver = nullptr;
    try {
        sv_ipc_receiver *const instance = new sv_ipc_receiver(socket_path);
        *receiver = instance;
    } catch (const std::bad_alloc &) {
        return SV_IPC_RECEIVER_STATUS_ALLOCATION_FAILED;
    } catch (...) {
        return SV_IPC_RECEIVER_STATUS_ALLOCATION_FAILED;
    }

    return SV_IPC_RECEIVER_STATUS_OK;
}

extern "C" sv_ipc_receiver_status sv_ipc_receiver_wait_for_sender(
    sv_ipc_receiver *receiver) {
    if (receiver == nullptr) {
        return SV_IPC_RECEIVER_STATUS_INVALID_ARGUMENT;
    }

    return receiver->receiver.wait_for_sender()
        ? SV_IPC_RECEIVER_STATUS_OK
        : SV_IPC_RECEIVER_STATUS_WAIT_FAILED;
}

extern "C" sv_ipc_receiver_status sv_ipc_receiver_receive_frame(
    sv_ipc_receiver *receiver,
    sv_ipc_receiver_host_frame *frame) {
    if (receiver == nullptr || frame == nullptr || frame->data != nullptr) {
        return SV_IPC_RECEIVER_STATUS_INVALID_ARGUMENT;
    }

    digiview_frame received;
    if (!receiver->receiver.receive_frame(received)) {
        return SV_IPC_RECEIVER_STATUS_RECEIVE_FAILED;
    }

    frame->data = reinterpret_cast<uint8_t *>(received.data);
    frame->timestamp = received.timestamp;
    frame->width = received.width;
    frame->height = received.height;
    frame->pixel_format = received.pixel_format;
    frame->pitch = received.pitch;
    return SV_IPC_RECEIVER_STATUS_OK;
}

extern "C" void sv_ipc_receiver_free_frame(sv_ipc_receiver_host_frame *frame) {
    if (frame == nullptr) {
        return;
    }

    std::free(frame->data);
    frame->data = nullptr;
}

extern "C" void sv_ipc_receiver_destroy(sv_ipc_receiver *receiver) {
    delete receiver;
}
