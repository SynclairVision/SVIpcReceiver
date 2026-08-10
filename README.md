# SVIpcReceiver

`SVIpcReceiver` is a standalone static library for receiving DigiView GPU IPC
frames. It exposes the `SVGpuIpcReceiver` class in `sv_ipc_receiver.hpp`.

## Requirements

The receiver uses DigiView's canonical IPC wire contract from its local
`digiview_commons` submodule:

```
digiview_commons/ipc_wire_structs.hpp
```

The library requires an installed CUDA Toolkit, located with CMake's
`find_package(CUDAToolkit REQUIRED)`; no CUDA or Jetson path is built into this
project.

For this receiver, `digiview_metadata` always uses the GPU width, height, and
flags wire layout. Consumers do not define or configure an IPC-layout macro.
Manual consumers must provide the receiver root and CUDA include paths and link
the CUDA runtime and driver libraries.

## Standalone build and continuous test

Initialize the local shared-contract submodule, then configure and build from
the `SVIpcReceiver` repository root:

```sh
git submodule update --init --recursive
cmake -S . -B build
cmake --build build
```

Configuration requires a CUDA Toolkit discoverable by CMake through
`find_package(CUDAToolkit REQUIRED)`. No Jetson-specific path is configured by
this project.

Run the continuous receiver test with the sender's UNIX socket path, or omit it
to use the receiver API's default `/tmp/source_camera_0_socket`:

```sh
./build/test [--socket PATH] [--frame N]
```

The test connects once, then receives and prints every frame's zero-based index,
dimensions, pitch, and pixel format until receiving fails or the process is
interrupted. It frees each received host buffer after processing it. `--frame N`
writes only received frame `N` as `frame_N.png` in the current working
directory. PNG capture converts BGR8 to PNG RGB and BGRA8 to PNG RGBA, respects
the received pitch, and is performed only for the selected frame. The test
needs an active, compatible DigiView GPU IPC sender at that socket and a
working CUDA-capable DigiView/Jetson runtime; it cannot produce frames.

## Test C ABI bridge

The default build also produces the test-only shared-library target
`sv_ipc_receiver_test_bridge`, named `libsv_ipc_receiver_test_bridge.so` on
Linux. It lets `test.py` use the real C++ receiver through `ctypes`, rather than
reimplementing the CUDA IPC protocol. The Python test utility is invoked from
this repository root as:

```sh
python3 test.py [--socket PATH] [--frame N]
```

`--frame N` writes only zero-based received frame `N` as `frame_N.png` in the
current working directory. The utility requires the CMake-built bridge, a
CUDA-capable DigiView/Jetson runtime, and an active compatible sender; it
cannot produce frames by itself.

Its C ABI is declared by `sv_ipc_receiver_test_bridge.h`. The opaque
`sv_ipc_receiver` handle is created with
`sv_ipc_receiver_create(const char *socket_path, sv_ipc_receiver **receiver)`.
The remaining calls are
`sv_ipc_receiver_wait_for_sender(sv_ipc_receiver *)`,
`sv_ipc_receiver_receive_frame(sv_ipc_receiver *, sv_ipc_receiver_host_frame *)`,
`sv_ipc_receiver_free_frame(sv_ipc_receiver_host_frame *)`, and
`sv_ipc_receiver_destroy(sv_ipc_receiver *)`. The status-returning calls use
`SV_IPC_RECEIVER_STATUS_OK` (0),
`SV_IPC_RECEIVER_STATUS_INVALID_ARGUMENT` (1),
`SV_IPC_RECEIVER_STATUS_ALLOCATION_FAILED` (2),
`SV_IPC_RECEIVER_STATUS_WAIT_FAILED` (3), and
`SV_IPC_RECEIVER_STATUS_RECEIVE_FAILED` (4). Initialize the host frame's `data`
member to null before receiving and release it with
`sv_ipc_receiver_free_frame` after each successful receive.

The C-compatible host-frame layout is:

```c
typedef struct sv_ipc_receiver_host_frame {
    uint8_t *data;
    uint64_t timestamp;
    int32_t width;
    int32_t height;
    int32_t pixel_format;
    int32_t pitch;
} sv_ipc_receiver_host_frame;
```

## CMake integration

Add this repository as a subdirectory and link the namespaced target:

```cmake
add_subdirectory(external/SVIpcReceiver)

target_link_libraries(my_receiver PRIVATE SVIpcReceiver::sv_ipc_receiver)
```

The target publishes the receiver root as an include directory, so
`<digiview_commons/ipc_wire_structs.hpp>` resolves to its checked-out local
submodule.

For manual, non-CMake integration, add the receiver root as the include root,
and link the CUDA runtime and driver libraries. On a successful
`receive_frame`, the caller owns `digiview_frame::data` and must release it
with `std::free`.
