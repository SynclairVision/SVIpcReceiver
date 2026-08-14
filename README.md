# SVIpcReceiver

`SVIpcReceiver` is a static C++ library for receiving DigiView GPU IPC frames.
Its public API is `SVGpuIpcReceiver` in `sv_ipc_receiver.hpp`.

## Quick start

The receiver requires a CUDA Toolkit discoverable by CMake and the local
`digiview_commons` submodule. The optional test tools also require ZLIB. From
this repository root:

```sh
git submodule update --init --recursive
cmake -S . -B build -DSV_IPC_RECEIVER_BUILD_TEST_TOOLS=ON
cmake --build build --target sv_ipc_receiver_test sv_ipc_receiver_test_bridge
```

Connect to an active, compatible DigiView sender at the default UNIX socket:

```sh
./build/sv_ipc_receiver_test
```

The default socket path is `/tmp/source_camera_0_socket0`. Pass `--socket PATH`
to the test utility, or a socket path to the example, to use a different sender.
The test utility continuously prints frame metadata until interrupted.
`./build/sv_ipc_receiver_test --frame N` also writes zero-based frame `N` as
`frame_N.png`.

## NVUNIXFD consumer (Jetson NX only)

`SVGpuIpcReceiver` supports only the CUDA IPC protocol. Consume DigiView
NVUNIXFD output through NVIDIA GStreamer. It requires Jetson NVMM and the NVIDIA
DeepStream `nvunixfdsrc` plugin; preflight the plugin with:

```sh
gst-inspect-1.0 nvunixfdsrc
```

Enable one processed DigiView pipeline with `ipc_frame_write=0` and
`nvunixfd_output=true`. That pipeline provides one endpoint for one supported
logical consumer. Receive one frame with:

```sh
timeout --foreground 10s gst-launch-1.0 -v nvunixfdsrc socket-path="/tmp/<pipeline-name>_nvunixfd_socket" buffer-timestamp-copy=true num-buffers=1 ! fakesink sync=false
```

## Use from C++

Add the repository as a subdirectory and link the namespaced target:

```cmake
add_subdirectory(external/SVIpcReceiver)
target_link_libraries(my_receiver PRIVATE SVIpcReceiver::sv_ipc_receiver)
```

Receive one frame with the default socket path:

```cpp
#include "sv_ipc_receiver.hpp"

#include <cstdlib>
#include <memory>

SVGpuIpcReceiver receiver;
if (receiver.wait_for_sender()) {
    digiview_frame frame;
    if (receiver.receive_frame(frame)) {
        std::unique_ptr<Npp8u, decltype(&std::free)> data(frame.data, &std::free);
        // Process frame.data while data owns it.
    }
}
```

Construct `SVGpuIpcReceiver` with a socket path to override the default. On a
successful `receive_frame`, the caller owns `digiview_frame::data` and must
release it with `std::free`.

## Build the example

`examples/receive_frame.cpp` receives and reports one frame, then exits. Build
it when needed without changing the default library build:

```sh
cmake -S . -B build -DSV_IPC_RECEIVER_BUILD_EXAMPLE=ON
cmake --build build --target receive_frame_example
./build/receive_frame_example [socket-path]
```

## Test bridge

With `SV_IPC_RECEIVER_BUILD_TEST_TOOLS=ON`, CMake builds the test-only
`libsv_ipc_receiver_test_bridge.so` for `test.py`. It exposes the real C++
receiver through `ctypes`; it is not required for C++ integration. Run it with:

```sh
python3 test.py [--socket PATH] [--frame N]
```

The C ABI and host-frame ownership contract are declared in
`sv_ipc_receiver_test_bridge.h`.

## Integration notes

The CMake target publishes this repository root as an include directory, so its
checked-out `digiview_commons/ipc_wire_structs.hpp` contract is available to
consumers. Manual integrations must add the receiver root and CUDA include
paths, then link the CUDA runtime and driver libraries.
