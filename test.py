"""Receive DigiView IPC frames through the test-only C ABI bridge."""

import argparse
import ctypes
from pathlib import Path
import signal
import struct
import sys
import zlib


DEFAULT_SOCKET_PATH = "/tmp/source_camera_0_socket0"
BRIDGE_LIBRARY_NAME = "libsv_ipc_receiver_test_bridge.so"
STATUS_OK = 0
PIXEL_FORMAT_BGR8 = 1
PIXEL_FORMAT_BGRA8 = 2
MAX_PNG_DIMENSION = (1 << 32) - 1

STATUS_NAMES = {
    0: "OK",
    1: "INVALID_ARGUMENT",
    2: "ALLOCATION_FAILED",
    3: "WAIT_FAILED",
    4: "RECEIVE_FAILED",
}


class SvIpcReceiver(ctypes.Structure):
    """Opaque receiver handle owned by the C ABI bridge."""


ReceiverPointer = ctypes.POINTER(SvIpcReceiver)


class HostFrame(ctypes.Structure):
    """Host frame layout defined by sv_ipc_receiver_test_bridge.h."""

    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("timestamp", ctypes.c_uint64),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("pixel_format", ctypes.c_int32),
        ("pitch", ctypes.c_int32),
    ]


def non_negative_frame(value):
    """Parse a non-negative frame index for argparse."""
    try:
        frame_number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "frame must be a non-negative integer"
        ) from error

    if frame_number < 0:
        raise argparse.ArgumentTypeError(
            "frame must be a non-negative integer"
        )
    return frame_number


def parse_arguments():
    """Parse command-line options."""
    script_directory = Path(__file__).resolve().parent
    default_library = script_directory / "build" / BRIDGE_LIBRARY_NAME
    parser = argparse.ArgumentParser(
        description="Receive frames with the SVIpcReceiver C ABI bridge."
    )
    parser.add_argument(
        "--socket",
        default=DEFAULT_SOCKET_PATH,
        metavar="PATH",
        help=f"sender UNIX socket (default: {DEFAULT_SOCKET_PATH})",
    )
    parser.add_argument(
        "--frame",
        type=non_negative_frame,
        metavar="N",
        help="zero-based frame index to save as frame_N.png",
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=default_library,
        metavar="PATH",
        help="test bridge shared library path",
    )
    return parser.parse_args()


def status_name(status):
    """Return a readable bridge status name."""
    return STATUS_NAMES.get(status, f"UNKNOWN_STATUS_{status}")


def load_bridge(library_path):
    """Load the bridge library and configure its C ABI signatures."""
    library_path = library_path.expanduser()
    if not library_path.is_file():
        raise RuntimeError(
            f"Bridge library not found: {library_path}. Build "
            "libsv_ipc_receiver_test_bridge.so with CMake first."
        )

    try:
        bridge = ctypes.CDLL(str(library_path))
    except OSError as error:
        raise RuntimeError(
            f"Unable to load bridge library {library_path}: {error}"
        ) from error

    try:
        bridge.sv_ipc_receiver_create.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ReceiverPointer),
        ]
        bridge.sv_ipc_receiver_create.restype = ctypes.c_int
        bridge.sv_ipc_receiver_wait_for_sender.argtypes = [ReceiverPointer]
        bridge.sv_ipc_receiver_wait_for_sender.restype = ctypes.c_int
        bridge.sv_ipc_receiver_receive_frame.argtypes = [
            ReceiverPointer,
            ctypes.POINTER(HostFrame),
        ]
        bridge.sv_ipc_receiver_receive_frame.restype = ctypes.c_int
        bridge.sv_ipc_receiver_free_frame.argtypes = [ctypes.POINTER(HostFrame)]
        bridge.sv_ipc_receiver_free_frame.restype = None
        bridge.sv_ipc_receiver_destroy.argtypes = [ReceiverPointer]
        bridge.sv_ipc_receiver_destroy.restype = None
    except AttributeError as error:
        raise RuntimeError(
            f"Bridge library {library_path} does not provide the required API: "
            f"{error}"
        ) from error

    return bridge


def frame_format(pixel_format):
    """Return the PNG channel count and color type for a frame format."""
    if pixel_format == PIXEL_FORMAT_BGR8:
        return 3, 2, "BGR8"
    if pixel_format == PIXEL_FORMAT_BGRA8:
        return 4, 6, "BGRA8"
    raise ValueError(f"unsupported pixel format: {pixel_format}")


def png_chunk(chunk_type, data):
    """Create one PNG chunk including its checksum."""
    if len(data) > MAX_PNG_DIMENSION:
        raise ValueError("PNG chunk is too large")

    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(data, checksum) & MAX_PNG_DIMENSION
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", checksum)
    )


def write_png(frame, output_path):
    """Write a selected BGR8 or BGRA8 host frame as a PNG file."""
    if not frame.data:
        raise ValueError("frame data is null")
    if frame.width <= 0 or frame.height <= 0 or frame.pitch <= 0:
        raise ValueError("frame dimensions and pitch must be positive")

    channels, color_type, _ = frame_format(frame.pixel_format)
    width = frame.width
    height = frame.height
    pitch = frame.pitch
    row_bytes = width * channels
    source_size = (height - 1) * pitch + row_bytes
    scanline_bytes = row_bytes + 1
    raw_size = height * scanline_bytes

    if width > MAX_PNG_DIMENSION or height > MAX_PNG_DIMENSION:
        raise ValueError("frame dimensions exceed PNG limits")
    if pitch < row_bytes:
        raise ValueError("frame pitch is smaller than its pixel row")
    if row_bytes > sys.maxsize:
        raise ValueError("frame row size is too large")
    if source_size > sys.maxsize:
        raise ValueError("frame buffer size is too large")
    if scanline_bytes > sys.maxsize or raw_size > sys.maxsize:
        raise ValueError("PNG scanline buffer is too large")

    source = ctypes.string_at(frame.data, source_size)
    raw = bytearray(raw_size)
    destination_offset = 0
    for row in range(height):
        raw[destination_offset] = 0
        destination_offset += 1
        source_offset = row * pitch
        for column in range(width):
            pixel_offset = source_offset + column * channels
            raw[destination_offset] = source[pixel_offset + 2]
            raw[destination_offset + 1] = source[pixel_offset + 1]
            raw[destination_offset + 2] = source[pixel_offset]
            if channels == 4:
                raw[destination_offset + 3] = source[pixel_offset + 3]
            destination_offset += channels

    compressed = zlib.compress(raw)
    ihdr = struct.pack(
        ">IIBBBBB", width, height, 8, color_type, 0, 0, 0
    )
    png_data = b"\x89PNG\r\n\x1a\n"
    png_data += png_chunk(b"IHDR", ihdr)
    png_data += png_chunk(b"IDAT", compressed)
    png_data += png_chunk(b"IEND", b"")
    output_path.write_bytes(png_data)


def print_frame(frame_index, frame):
    """Print metadata for one received frame."""
    try:
        _, _, pixel_format = frame_format(frame.pixel_format)
    except ValueError:
        pixel_format = f"unknown ({frame.pixel_format})"

    print(
        f"Frame {frame_index}: {frame.width}x{frame.height}, "
        f"pitch {frame.pitch}, pixel format {pixel_format}, "
        f"timestamp {frame.timestamp}"
    )


def handle_shutdown_signal(_signum, _frame):
    """Stop receiving so the existing cleanup paths can run."""
    raise KeyboardInterrupt


def receive_frames(options):
    """Create a receiver, wait once, and continuously receive frames."""
    receiver = ReceiverPointer()

    try:
        bridge = load_bridge(options.library)
        socket_path = options.socket.encode("utf-8")
        status = bridge.sv_ipc_receiver_create(
            socket_path, ctypes.byref(receiver)
        )
        if status != STATUS_OK:
            print(
                f"Receiver creation failed: {status_name(status)} ({status}).",
                file=sys.stderr,
            )
            return 1
        if not receiver:
            print(
                "Receiver creation failed: bridge returned a null receiver.",
                file=sys.stderr,
            )
            return 1

        status = bridge.sv_ipc_receiver_wait_for_sender(receiver)
        if status != STATUS_OK:
            print(
                f"Waiting for sender failed: {status_name(status)} ({status}).",
                file=sys.stderr,
            )
            return 1

        frame_index = 0
        frame_saved = False
        while True:
            frame = HostFrame()
            try:
                status = bridge.sv_ipc_receiver_receive_frame(
                    receiver, ctypes.byref(frame)
                )
                if status != STATUS_OK:
                    print(
                        f"Frame receive failed: {status_name(status)} "
                        f"({status}).",
                        file=sys.stderr,
                    )
                    return 1

                print_frame(frame_index, frame)
                if options.frame == frame_index and not frame_saved:
                    output_path = Path.cwd() / f"frame_{frame_index}.png"
                    write_png(frame, output_path)
                    frame_saved = True
                    print(f"Saved {output_path.name}.")

                frame_index += 1
            finally:
                bridge.sv_ipc_receiver_free_frame(ctypes.byref(frame))
    except KeyboardInterrupt:
        print("Receive interrupted.", file=sys.stderr)
        return 0
    finally:
        if receiver:
            bridge.sv_ipc_receiver_destroy(receiver)


def main():
    """Run the command-line receiver utility."""
    options = parse_arguments()
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    # Let signals interrupt ctypes calls blocked in the receiver.
    signal.siginterrupt(signal.SIGINT, True)
    signal.siginterrupt(signal.SIGTERM, True)
    try:
        return receive_frames(options)
    except (OSError, RuntimeError, UnicodeError, ValueError, zlib.error) as error:
        print(f"Test utility failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
