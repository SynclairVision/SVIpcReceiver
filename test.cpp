#include "sv_ipc_receiver.hpp"

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <signal.h>
#include <string>
#include <vector>

#include <zlib.h>

namespace {

const char kDefaultSocketPath[] = "/tmp/source_camera_0_socket0";
volatile std::sig_atomic_t g_interrupted = 0;

struct Options {
    std::string socket_path = kDefaultSocketPath;
    bool save_frame = false;
    std::uint64_t frame_number = 0;
};

void handle_signal(int) {
    g_interrupted = 1;
}

void print_usage(const char *program) {
    std::cerr << "Usage: " << program
              << " [--socket PATH] [--frame N]\n"
                 "Default socket path: "
              << kDefaultSocketPath << '\n';
}

bool parse_frame_number(const char *text, std::uint64_t *frame_number) {
    if (text == nullptr || text[0] == '\0' || text[0] == '-' || frame_number == nullptr) {
        return false;
    }

    errno = 0;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        return false;
    }

    if (value > std::numeric_limits<std::uint64_t>::max()) {
        return false;
    }

    *frame_number = static_cast<std::uint64_t>(value);
    return true;
}

bool parse_options(int argc, char *argv[], Options *options) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--socket" && index + 1 < argc) {
            options->socket_path = argv[++index];
        } else if (argument == "--frame" && index + 1 < argc) {
            if (!parse_frame_number(argv[++index], &options->frame_number)) {
                return false;
            }
            options->save_frame = true;
        } else {
            return false;
        }
    }

    return !options->socket_path.empty();
}

void write_big_endian(std::ofstream *output, const std::uint32_t value) {
    const unsigned char bytes[] = {
        static_cast<unsigned char>((value >> 24U) & 0xFFU),
        static_cast<unsigned char>((value >> 16U) & 0xFFU),
        static_cast<unsigned char>((value >> 8U) & 0xFFU),
        static_cast<unsigned char>(value & 0xFFU)};
    output->write(reinterpret_cast<const char *>(bytes), sizeof(bytes));
}

bool write_chunk(
    std::ofstream *output,
    const char type[4],
    const unsigned char *data,
    const std::size_t data_size) {
    if (data_size > std::numeric_limits<std::uint32_t>::max() ||
        data_size > std::numeric_limits<uInt>::max() ||
        data_size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }

    write_big_endian(output, static_cast<std::uint32_t>(data_size));
    output->write(type, 4);
    if (data_size != 0U) {
        output->write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(data_size));
    }

    uLong crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, reinterpret_cast<const Bytef *>(type), 4);
    if (data_size != 0U) {
        crc = crc32(crc, reinterpret_cast<const Bytef *>(data), static_cast<uInt>(data_size));
    }
    write_big_endian(output, static_cast<std::uint32_t>(crc));
    return output->good();
}

bool write_png(const digiview_frame &frame, const std::string &filename) {
    if (frame.data == nullptr || frame.width <= 0 || frame.height <= 0 || frame.pitch <= 0) {
        std::cerr << "Cannot save an invalid frame.\n";
        return false;
    }

    const ipc::GpuFramePixelFormat pixel_format =
        static_cast<ipc::GpuFramePixelFormat>(frame.pixel_format);
    const std::size_t bytes_per_pixel = ipc::gpu_frame_bytes_per_pixel(pixel_format);
    if (bytes_per_pixel == 0U) {
        std::cerr << "Cannot save an unsupported pixel format.\n";
        return false;
    }

    const std::size_t width = static_cast<std::size_t>(frame.width);
    const std::size_t height = static_cast<std::size_t>(frame.height);
    const std::size_t pitch = static_cast<std::size_t>(frame.pitch);
    if (width > std::numeric_limits<std::size_t>::max() / bytes_per_pixel) {
        std::cerr << "Frame row size overflows.\n";
        return false;
    }
    const std::size_t row_bytes = width * bytes_per_pixel;
    if (pitch < row_bytes ||
        (height > 1U && height - 1U >
            (std::numeric_limits<std::size_t>::max() - row_bytes) / pitch)) {
        std::cerr << "Frame pitch is invalid.\n";
        return false;
    }

    if (row_bytes == std::numeric_limits<std::size_t>::max() ||
        height > std::numeric_limits<std::size_t>::max() / (row_bytes + 1U)) {
        std::cerr << "PNG scanline size overflows.\n";
        return false;
    }
    const std::size_t scanline_bytes = row_bytes + 1U;
    const std::size_t raw_size = height * scanline_bytes;
    if (raw_size > static_cast<std::size_t>(std::numeric_limits<uLong>::max())) {
        std::cerr << "PNG input is too large for zlib.\n";
        return false;
    }

    std::vector<unsigned char> raw(raw_size);
    std::size_t destination_offset = 0;
    for (std::size_t y = 0; y < height; ++y) {
        raw[destination_offset++] = 0;
        const unsigned char *const source = reinterpret_cast<const unsigned char *>(frame.data) + y * pitch;
        for (std::size_t x = 0; x < width; ++x) {
            const unsigned char *const pixel = source + x * bytes_per_pixel;
            raw[destination_offset++] = pixel[2];
            raw[destination_offset++] = pixel[1];
            raw[destination_offset++] = pixel[0];
            if (pixel_format == ipc::GpuFramePixelFormat::kBgra8) {
                raw[destination_offset++] = pixel[3];
            }
        }
    }

    const uLong bound = compressBound(static_cast<uLong>(raw.size()));
    if (bound > static_cast<uLong>(std::numeric_limits<std::size_t>::max()) ||
        bound > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "Compressed PNG is too large.\n";
        return false;
    }
    std::vector<unsigned char> compressed(static_cast<std::size_t>(bound));
    uLongf compressed_size = bound;
    if (compress2(
            compressed.data(),
            &compressed_size,
            raw.data(),
            static_cast<uLong>(raw.size()),
            Z_BEST_SPEED) != Z_OK ||
        compressed_size > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "zlib failed to compress the PNG.\n";
        return false;
    }
    compressed.resize(static_cast<std::size_t>(compressed_size));

    std::ofstream output(filename.c_str(), std::ios::binary);
    if (!output) {
        std::cerr << "Cannot open " << filename << " for writing.\n";
        return false;
    }

    const unsigned char signature[] = {0x89U, 0x50U, 0x4EU, 0x47U, 0x0DU, 0x0AU, 0x1AU, 0x0AU};
    const unsigned char header[] = {
        static_cast<unsigned char>((width >> 24U) & 0xFFU),
        static_cast<unsigned char>((width >> 16U) & 0xFFU),
        static_cast<unsigned char>((width >> 8U) & 0xFFU),
        static_cast<unsigned char>(width & 0xFFU),
        static_cast<unsigned char>((height >> 24U) & 0xFFU),
        static_cast<unsigned char>((height >> 16U) & 0xFFU),
        static_cast<unsigned char>((height >> 8U) & 0xFFU),
        static_cast<unsigned char>(height & 0xFFU),
        8U,
        pixel_format == ipc::GpuFramePixelFormat::kBgr8 ? 2U : 6U,
        0U,
        0U,
        0U};
    output.write(reinterpret_cast<const char *>(signature), sizeof(signature));
    if (!write_chunk(&output, "IHDR", header, sizeof(header)) ||
        !write_chunk(&output, "IDAT", compressed.data(), compressed.size()) ||
        !write_chunk(&output, "IEND", nullptr, 0U)) {
        std::cerr << "Failed while writing " << filename << ".\n";
        return false;
    }

    return true;
}

const char *pixel_format_name(const int32_t pixel_format) {
    switch (static_cast<ipc::GpuFramePixelFormat>(pixel_format)) {
    case ipc::GpuFramePixelFormat::kBgr8:
        return "BGR8";
    case ipc::GpuFramePixelFormat::kBgra8:
        return "BGRA8";
    default:
        return "unsupported";
    }
}

} // namespace

int main(int argc, char *argv[]) {
    try {
        Options options;
        if (!parse_options(argc, argv, &options)) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }

        struct sigaction action {};
        action.sa_handler = handle_signal;
        sigemptyset(&action.sa_mask);
        action.sa_flags = 0;
        if (sigaction(SIGINT, &action, nullptr) != 0 || sigaction(SIGTERM, &action, nullptr) != 0) {
            std::cerr << "Unable to install signal handlers.\n";
            return EXIT_FAILURE;
        }

        SVGpuIpcReceiver receiver(options.socket_path);
        if (!receiver.wait_for_sender()) {
            if (g_interrupted != 0) {
                std::cerr << "Connection interrupted.\n";
                return EXIT_SUCCESS;
            }
            std::cerr << "Unable to connect to a compatible DigiView sender at "
                      << options.socket_path << ".\n";
            return EXIT_FAILURE;
        }

        std::uint64_t frame_index = 0;
        while (g_interrupted == 0) {
            digiview_frame frame;
            if (!receiver.receive_frame(frame)) {
                if (g_interrupted != 0) {
                    std::cerr << "Receive interrupted.\n";
                    return EXIT_SUCCESS;
                }
                std::cerr << "Frame receive failed.\n";
                return EXIT_FAILURE;
            }

            std::unique_ptr<Npp8u, decltype(&std::free)> frame_data(frame.data, &std::free);
            std::cout << "Frame " << frame_index << ": " << frame.width << "x" << frame.height
                      << ", pitch " << frame.pitch << ", pixel-format "
                      << pixel_format_name(frame.pixel_format) << '\n';

            if (options.save_frame && frame_index == options.frame_number) {
                const std::string filename = "frame_" + std::to_string(frame_index) + ".png";
                if (!write_png(frame, filename)) {
                    return EXIT_FAILURE;
                }
                std::cout << "Saved " << filename << ".\n";
            }

            if (frame_index == std::numeric_limits<std::uint64_t>::max()) {
                std::cerr << "Frame index overflow.\n";
                return EXIT_FAILURE;
            }
            ++frame_index;
        }

        std::cerr << "Receive interrupted.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "Test utility failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
