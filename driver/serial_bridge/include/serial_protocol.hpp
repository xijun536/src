#pragma once

#include <cstddef>
#include <cstdint>

namespace serial_bridge::protocol
{

// 帧头2 + 类型1 + 数据长度2 + 序号2 + CRC2 + 帧尾1

//帧头=0xAA+0x55,帧尾=0x4B
inline constexpr uint8_t kHeaderFirst = 0xAA;
inline constexpr uint8_t kHeaderSecond = 0x55;
inline constexpr uint8_t kFrameTail = 0x4B;

//各个部分的数据长度
inline constexpr std::size_t kHeaderSize = 2;
inline constexpr std::size_t kLengthFieldSize = 2;
inline constexpr std::size_t kSequenceFieldSize = 2;
inline constexpr std::size_t kCrcSize = 2;
inline constexpr std::size_t kTailSize = 1;

//除去数据本身的固定开销
inline constexpr std::size_t kFixedFrameSize =
  kHeaderSize +
  sizeof(uint8_t) +
  kLengthFieldSize +
  kSequenceFieldSize +
  kCrcSize +
  kTailSize;

//底盘速度、机械臂和机械臂状态消息的数据长度
inline constexpr std::size_t kBaseVelocityPayloadSize = 12; //三个float型，xyz
inline constexpr std::size_t kArmCommandPayloadSize = 16;  //任务id+三个float:z,x,yaw
inline constexpr std::size_t kArmStatusPayloadSize = 19;
inline constexpr std::size_t kMaxPayloadSize = 64;

//消息类型枚举，添加了类型保护
enum class MessageType : uint8_t
{
  kBaseVelocityCommand = 0x01,
  kArmPositionCommand = 0x02,
  kArmStatus = 0x82,
};

//机械臂命令状态枚举，添加了类型保护
enum class ArmState : uint8_t
{
  kIdle = 0,
  kAccepted = 1,
  kMoving = 2,
  kSucceeded = 3,
  kFailed = 4,
};

}  // namespace serial_bridge::protocol
