#include "sv_ipc_receiver.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char *argv[]) {
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0]
                  << " [socket-path]\n"
                     "Default socket path: /tmp/source_camera_0_socket\n";
        return EXIT_FAILURE;
    }

    const std::string socket_path = argc == 2 ? argv[1] : "/tmp/source_camera_0_socket";
    SVGpuIpcReceiver receiver(socket_path);
    if (!receiver.wait_for_sender()) {
        std::cerr << "Unable to connect to a compatible DigiView sender at "
                  << socket_path << ".\n";
        return EXIT_FAILURE;
    }

    digiview_frame frame;
    if (!receiver.receive_frame(frame)) {
        std::cerr << "Failed to receive a frame from " << socket_path << ".\n";
        return EXIT_FAILURE;
    }

    std::unique_ptr<Npp8u, decltype(&std::free)> frame_data(frame.data, &std::free);
    if (frame.data == nullptr || frame.width <= 0 || frame.height <= 0 || frame.pitch <= 0) {
        std::cerr << "Received an invalid frame.\n";
        return EXIT_FAILURE;
    }

    std::cout << "Received " << frame.width << "x" << frame.height
              << " frame (pitch " << frame.pitch << ", timestamp "
              << frame.timestamp << ").\n";
    return EXIT_SUCCESS;
}
