#include "transport.hpp"

namespace serial_bridge
{
    SerialPort::SerialPort(const std::string &device_name,
                           uint32_t baud_rate,
                           std::function<void(const uint8_t *, std::size_t)> callback): 
                           streamflow_(std::move(callback))

    {
        try
        {
            drivers::serial_driver::SerialPortConfig config(
                baud_rate,
                drivers::serial_driver::FlowControl::NONE,
                drivers::serial_driver::Parity::NONE,
                drivers::serial_driver::StopBits::ONE);
            // 子线程
            io_context_ = std::make_shared<drivers::common::IoContext>(1);
            // 赋值为可操作串口对象
            serial_driver_ = std::make_shared<drivers::serial_driver::SerialDriver>(*io_context_);
            serial_driver_->init_port(device_name, config);
            serial_driver_->port()->open();
        }
        catch (const std::exception &error)
        {
            throw std::runtime_error(std::string("fail to open: ") + error.what());
        }
        running_.store(true);
        usb_stream_receive();
    }
    SerialPort::~SerialPort()
    {
      running_.store(false);
      if (serial_driver_ && serial_driver_->port() && serial_driver_->port()->is_open())
      {
        serial_driver_->port()->close();
      }
      if (io_context_)
      {
        io_context_->waitForExit();
      }
    }

    void SerialPort::usb_stream_receive()
    {
        if (!running_ || !serial_driver_->port()->is_open())
        {
            return;
        }
         // 持续接受
        serial_driver_->port()->async_receive(
         // 主线程创建了Lambda并且，在async_receive注册
        [this](std::vector<uint8_t> &bytes, const std::size_t &size)
        {
          // 不由主线程执行
          // 等串口收到数据后，由 IoContext 的 I/O 子线程执行
          if (!running_)
             return;
          streamflow_(bytes.data(), size);
          usb_stream_receive();
        });
    }
    std::vector<uint8_t> build_frame(const std::vector<uint8_t> &data)
    {
      std::vector<uint8_t> frame;
      //0xA5+data.size()+0x4B
      frame.reserve(1 + data.size() + 1);

      frame.push_back(0xA5);
      // data
      frame.insert(frame.end(), data.begin(), data.end());

      frame.push_back(0x4B);
      return frame;
    }
}