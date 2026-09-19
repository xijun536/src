#pragma once
#include <serial_driver/serial_driver.hpp>

#include <vector>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <functional>
#include <stdexcept>
#include <cstring>
#include <atomic>

namespace serial_bridge
{
    struct NavPacket
    {
      float linear_x = 0.0;
      float linear_y = 0.0;
      float angular_z = 0.0;
    }; 

    class SerialPort
    {
    public:
        SerialPort(const std::string & device_name,
              uint32_t baud_rate,
              std::function<void(const uint8_t*, std::size_t)> callback);
        ~SerialPort();
    std::shared_ptr<drivers::serial_driver::SerialDriver> serial_driver_;
    private:
    void usb_stream_receive();
    std::function<void(const uint8_t *, std::size_t)> streamflow_;
    std::shared_ptr<drivers::common::IoContext> io_context_;
    std::atomic_bool running_{false};
    };

    std::vector<uint8_t> build_frame(const std::vector<uint8_t> &data); 
}