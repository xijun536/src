#pragma once
#include <rclcpp/rclcpp.hpp>
#include "transport.hpp"
#include <geometry_msgs/msg/twist.hpp>

#include <memory>
#include <mutex>
#include <vector>
#include <chrono>
#include <cstring>
#include <utility>

namespace serial_bridge
{
    class DriverNode : public rclcpp::Node
    {
    public:
      explicit DriverNode(const rclcpp::NodeOptions &options);
      ~DriverNode() override;

    private:
       std::chrono::steady_clock::time_point last_cmd_vel_time_{};
       std::unique_ptr<SerialPort> serial_;
       rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr nav_sub_;
       rclcpp::TimerBase::SharedPtr nav_timer_;
       NavPacket nav_packet_;
       std::mutex nav_mutex_;  
    };
}