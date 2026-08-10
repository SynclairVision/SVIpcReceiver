#ifndef SV_IPC_RECEIVER_TEST_BRIDGE_H
#define SV_IPC_RECEIVER_TEST_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sv_ipc_receiver sv_ipc_receiver;

typedef enum sv_ipc_receiver_status {
    SV_IPC_RECEIVER_STATUS_OK = 0,
    SV_IPC_RECEIVER_STATUS_INVALID_ARGUMENT = 1,
    SV_IPC_RECEIVER_STATUS_ALLOCATION_FAILED = 2,
    SV_IPC_RECEIVER_STATUS_WAIT_FAILED = 3,
    SV_IPC_RECEIVER_STATUS_RECEIVE_FAILED = 4
} sv_ipc_receiver_status;

typedef struct sv_ipc_receiver_host_frame {
    uint8_t *data;
    uint64_t timestamp;
    int32_t width;
    int32_t height;
    int32_t pixel_format;
    int32_t pitch;
} sv_ipc_receiver_host_frame;

sv_ipc_receiver_status sv_ipc_receiver_create(
    const char *socket_path,
    sv_ipc_receiver **receiver);
sv_ipc_receiver_status sv_ipc_receiver_wait_for_sender(sv_ipc_receiver *receiver);
sv_ipc_receiver_status sv_ipc_receiver_receive_frame(
    sv_ipc_receiver *receiver,
    sv_ipc_receiver_host_frame *frame);
void sv_ipc_receiver_free_frame(sv_ipc_receiver_host_frame *frame);
void sv_ipc_receiver_destroy(sv_ipc_receiver *receiver);

#ifdef __cplusplus
}
#endif

#endif
