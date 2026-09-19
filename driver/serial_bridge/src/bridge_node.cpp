#include "bridge_node.hpp"

namespace serial_bridge
{
  //NodeOptions 不是节点名称 是保存参数覆盖、话题重映射、ROS 命令行参数等节点创建选项
    DriverNode::DriverNode(const rclcpp::NodeOptions &options) : Node("serial_driver", options)
    {
        auto device_name = declare_parameter<std::string>("config.device_name");
        auto baud_rate = declare_parameter<int>("config.baud_rate");
        auto cmd_vel_topic = declare_parameter<std::string>("topic.cmd_vel");
        auto interval = declare_parameter<int>("interval");

        serial_ = std::make_unique<SerialPort>(device_name,static_cast<uint32_t>(baud_rate),
        [](const uint8_t *, std::size_t)
        {
          //接收单片机的信息处理
            // decoder_持续增流
          /*  std::lock_guard<std::mutex> lock(decoder_mutex_);
            decoder_.append(data, size);*/
        });

        nav_sub_ = create_subscription<geometry_msgs::msg::Twist>(cmd_vel_topic,
            10, [this](const geometry_msgs::msg::Twist::SharedPtr msg)
            {
          try
          {
            std::scoped_lock lock{nav_mutex_};
            nav_packet_.linear_x = static_cast<float>(msg->linear.x);
            nav_packet_.linear_y = static_cast<float>(msg->linear.y);
            nav_packet_.angular_z = static_cast<float>(msg->angular.z);
            last_cmd_vel_time_ = std::chrono::steady_clock::now();
          }
          catch (const std::exception& error)
          {
            RCLCPP_ERROR(get_logger(), "Error from nav callback: %s", error.what());
          } });

        nav_timer_ = create_wall_timer(std::chrono::milliseconds(interval),
        [this]()
        {
            NavPacket packet;
            std::chrono::steady_clock::time_point last_time_;
            {
              std::scoped_lock lock{nav_mutex_};
              packet = nav_packet_;
              last_time_ = last_cmd_vel_time_;
            }
            const auto now = std::chrono::steady_clock::now();
            if(now - last_time_>std::chrono::milliseconds(200))
            {
              packet = NavPacket{};
            }
            std::vector<uint8_t> data(sizeof(packet));
            std::memcpy(data.data(), &packet, sizeof(packet));
            auto frame = build_frame(std::move(data));
            serial_->serial_driver_->port()->send(frame);
        });
    }
    DriverNode::~DriverNode() = default;
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<serial_bridge::DriverNode>(rclcpp::NodeOptions{});
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}