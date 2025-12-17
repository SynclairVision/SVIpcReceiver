# SVIpcReceiver
Contains code for communicating with DigiView via IPC

## Installation
To use the SVIpcReceiver class in your project, include the `sv_ipc_receiver.hpp` header file and add the `sv_ipc_receiver.cpp` source file to your project.

Small CMake example:

```cmake
add_executable(my_receiver main.cpp sv_ipc_receiver.cpp)
target_link_libraries(my_receiver nvbufsurface cudart)
```

## Usage
The class returns a `digiview_frame` struct containing the frame data and metadata, where the frame data is a CUDA pointer:

```cpp
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
```

Here is a simple example of how to use the SVIpcReceiver class:

```cpp
#include "sv_ipc_receiver.hpp"

int main() {
    int32_t f_idx = 0;

    SVIpcReceiver receiver("/tmp/source_camera_0_socket");
    if (!receiver.wait_for_sender()) {
        return -1;
    }

    digiview_frame frame;
    bool last_frame = false;

    while (!last_frame) {
        if (!receiver.receive_frame(frame)) {
            break;
        }

        // Print frame metadata
        printf("Frame %d:\n", f_idx);
        printf("  Timestamp: %lu\n", frame.timestamp);
        printf("  Acceleration: [%f, %f, %f]\n", frame.acc[0], frame.acc[1], frame.acc[2]);
        printf("  Velocity: [%f, %f, %f]\n", frame.vel[0], frame.vel[1], frame.vel[2]);
        printf("  Direction: [%f, %f, %f]\n", frame.dir[0], frame.dir[1], frame.dir[2]);
        printf("  Width: %d\n", frame.width);
        printf("  Height: %d\n", frame.height);
        printf("  Pixel Format: %d\n", frame.pixel_format);
        printf("  Pitch: %d\n", frame.pitch);
        
        f_idx++;

        // Now the frame is accessible via frame.data pointer.
    }

    receiver.cleanup();
    return 0;
}
```

## Wrapping in python (UNTESTED)
To wrap the SVIpcReceiver class in Python, you can use a library like pybind11 or ctypes. Below is an example using pybind11:

```cpp
#include <pybind11/pybind11.h>
#include "sv_ipc_receiver.hpp"
namespace py = pybind11;
PYBIND11_MODULE(sv_ipc_receiver) {
    py::class_<SVIpcReceiver>("SVIpcReceiver")
        .def(py::init<const std::string&>())
        .def("wait_for_sender", &SVIpcReceiver::wait_for_sender)
        .def("receive_frame", &SVIpcReceiver::receive_frame)
        .def("cleanup", &SVIpcReceiver::cleanup);
}
```

In python:
```python
from sv_ipc_receiver import SVIpcReceiver

rx = SVIpcReceiver("/tmp/source_camera_0_socket") 

rx.wait_for_sender()

frame = {}
while rx.receive_frame(frame):
    print("ts:", frame["timestamp"])
rx.cleanup()
```
