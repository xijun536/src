# D435i Vision：ROS 2 Humble 视觉模式节点

`d435i_vision` 是一个标准 ROS 2 `ament_python` 包。节点只打开一次 Intel
RealSense D435i，并通过 ROS 2 消息在待机、圆形上表面跟踪和 RGB 色环检测之间
切换。强类型状态由配套的 `cv3_interfaces` 包提供。

## 环境

- ROS 2 Humble；
- Python 3.10 的 Conda 环境 `d435i`；
- OpenCV、NumPy、PyYAML 和 `pyrealsense2`；
- 与系统匹配的 RealSense SDK、设备权限和 USB 3 连接（仅硬件运行需要）。

每个新终端都必须分别加载 ROS 2 Humble 和本工作区：

```bash
source /opt/ros/humble/setup.bash
source /home/book/Desktop/install/setup.bash
```

两个 `source` 都只对当前终端生效。下文出现“已完整加载环境的终端”时，均指该
终端已经执行了这两行；需要运行视觉节点时还要激活 `d435i` Conda 环境。

## 标准构建与启动

在具备正常 ROS 2 Python 构建依赖的环境中，标准单命令工作流是：

```bash
source /opt/ros/humble/setup.bash
conda activate d435i
cd /home/book/Desktop
colcon build --symlink-install \
  --packages-select cv3_interfaces d435i_vision
source /home/book/Desktop/install/setup.bash
ros2 launch d435i_vision d435i.launch.py
```

本机 `/home/book/Desktop` 还保留了旧项目内的 `cv3_interfaces`。为避免 colcon
同时发现两个同名包，确定性构建必须增加
`--base-paths /home/book/Desktop/src/d435i`。此外，本机的 `d435i` Conda 环境缺少
ROS 2 接口生成器需要的 `em`，因此本机不要直接重复上面的单命令构建，也不要
通过网络临时安装依赖；请使用下一节已经验证的分阶段构建。

## 本机已验证的确定性构建

先在一个新终端中加载 Humble 和已有工作区环境，并用受控的系统 PATH 重建接口：

```bash
source /opt/ros/humble/setup.bash
source /home/book/Desktop/install/setup.bash
cd /home/book/Desktop
env PATH=/usr/bin:/bin:/opt/ros/humble/bin \
  colcon build --symlink-install \
    --base-paths /home/book/Desktop/src/d435i \
    --packages-select cv3_interfaces --cmake-clean-cache
```

然后在另一个新终端中再次加载 Humble 和工作区，激活 Conda，并保持下面已经验证
过的 `PYTHONPATH` 顺序来构建 Python 运行包：

```bash
source /opt/ros/humble/setup.bash
source /home/book/Desktop/install/setup.bash
source /home/book/miniconda3/etc/profile.d/conda.sh
conda activate d435i
cd /home/book/Desktop
PYTHONPATH=/home/book/miniconda3/envs/d435i/lib/python3.10/site-packages:/usr/lib/python3/dist-packages:${PYTHONPATH} \
  /home/book/miniconda3/envs/d435i/bin/python /usr/bin/colcon build \
    --symlink-install \
    --base-paths /home/book/Desktop/src/d435i \
    --packages-select d435i_vision
source /home/book/Desktop/install/setup.bash
ros2 launch d435i_vision d435i.launch.py
```

构建完成后，每个新终端仍须重新加载 Humble 和
`/home/book/Desktop/install/setup.bash`。

## 启动选项

默认启动使用已安装的 `config/config.yaml`。无桌面环境可在已完整加载环境并激活
`d435i` 的终端中运行：

```bash
ros2 launch d435i_vision d435i.launch.py no_display:=true
```

使用其他配置文件：

```bash
ros2 launch d435i_vision d435i.launch.py config_file:=/absolute/path/to/config.yaml
```

查看节点命令行参数（不会打开相机）：

```bash
ros2 run d435i_vision d435i_vision --help
```

离线模式不是命令行回退：必须在 YAML 中设置 `ros.enabled: false`。此时程序不
初始化 ROS，并固定使用 `initial_mode`；`no_display:=true` 只关闭 OpenCV GUI，
不会自动关闭 ROS。

## ROS 控制与检查

在另一个已完整加载环境的终端中，用 `UInt8` 切换模式：

```bash
source /opt/ros/humble/setup.bash
source /home/book/Desktop/install/setup.bash
ros2 topic pub --once /vision_mode std_msgs/msg/UInt8 "{data: 0}"
ros2 topic pub --once /vision_mode std_msgs/msg/UInt8 "{data: 1}"
ros2 topic pub --once /vision_mode std_msgs/msg/UInt8 "{data: 2}"
```

默认值含义如下：

| 值 | 模式 | 行为 |
|---:|---|---|
| 0 | `idle` | 相机继续采集，不运行视觉算法 |
| 1 | `surface_tracking` | RGB、深度和可选 IMU 的上表面检测与跟踪 |
| 2 | `ring_detection` | 使用共享 RGB 帧检测色环 |

未映射的值只产生警告，当前模式不变。模式切换不会重启相机，有效切换会清除
表面跟踪器的旧状态。

模式 1 的目标颜色通过独立话题切换：

```bash
ros2 topic pub --once /vision_color std_msgs/msg/String "{data: yellow}"
```

可用颜色由 YAML 的 `colors` 配置决定；非法名称只产生警告并保留当前颜色。

查看每个成功处理帧发布的强类型状态：

```bash
ros2 topic echo --qos-reliability best_effort /vision/status
```

查询内存中的最新状态（不会重新读取相机或运行算法）：

```bash
ros2 service call /vision/get_status cv3_interfaces/srv/GetVisionStatus "{}"
```

`cache_available` 表示当前控制状态下是否已有完整帧，`result_age_s` 是缓存年龄。
模式或颜色改变后旧缓存立即失效，下一帧完成前会返回
`cache_available=false`、`result_available=false`。

## 配置与输出

默认 ROS 配置包括节点名 `cv3_vision_controller`、控制话题 `/vision_mode` 和
`/vision_color`、状态话题 `/vision/status`、查询服务 `/vision/get_status`，以及
默认关闭的兼容 JSON 结果话题 `/vision/result`。实际运行行为以安装的 YAML 为准。

检测结果以紧凑 JSON 写到终端。模式 1 保留表面跟踪字段；模式 2 返回零个或任意
数量的有效椭圆，按 X 坐标从左到右排序。任意正数色环均视为成功；只有恰好三个
色环时才增加中间色环中心和左右色环 Y 差字段。启用
`ros.result.enabled` 后，`/vision/result` 发布与终端相同的 JSON。

启用显示时，主窗口 `vision-main` 始终保留：`idle` 有 1 个窗口，
`surface_tracking` 有 3 个窗口，`ring_detection` 有 4 个窗口。按 `q` 或 Esc
退出显示模式。

## 自动化验证

以下检查不连接 D435i，也不启动相机。先在当前终端加载 Humble 和工作区：

```bash
source /opt/ros/humble/setup.bash
source /home/book/Desktop/install/setup.bash
cd /home/book/Desktop/src/d435i/d435i_vision
PYTHONDONTWRITEBYTECODE=1 /home/book/miniconda3/envs/d435i/bin/python \
  -m pytest -q -p no:cacheprovider test
PYTHONPYCACHEPREFIX=/tmp/d435i-final-pycache \
  /home/book/miniconda3/envs/d435i/bin/python -m compileall -q -f \
  d435i_vision launch
```

安装发现检查也必须在已加载 Humble 和工作区的终端中执行：

```bash
source /opt/ros/humble/setup.bash
source /home/book/Desktop/install/setup.bash
ros2 pkg prefix cv3_interfaces
ros2 pkg prefix d435i_vision
ros2 interface show cv3_interfaces/srv/GetVisionStatus
ros2 run d435i_vision d435i_vision --help
ROS_LOG_DIR=/home/book/Desktop/log/ros \
  ros2 launch d435i_vision d435i.launch.py --show-args
```

如果出现 `ModuleNotFoundError: cv3_interfaces`，说明工作区尚未成功构建，或者当前
终端没有执行 `source /home/book/Desktop/install/setup.bash`。按“本机已验证的确定
性构建”重建两个包，并在当前终端重新加载 Humble 和工作区后再试。

## D435i 硬件验收（待人工执行）

以下命令只是人工验收说明，本次迁移没有连接或运行 D435i，不能据此声称相机、
窗口或模式行为已经通过。连接 D435i 后，在第一个新终端中运行：

```bash
source /opt/ros/humble/setup.bash
source /home/book/Desktop/install/setup.bash
source /home/book/miniconda3/etc/profile.d/conda.sh
conda activate d435i
ros2 launch d435i_vision d435i.launch.py
```

在第二个新终端中完整加载相同环境后运行：

```bash
source /opt/ros/humble/setup.bash
source /home/book/Desktop/install/setup.bash
source /home/book/miniconda3/etc/profile.d/conda.sh
conda activate d435i
ros2 topic pub --once /vision_mode std_msgs/msg/UInt8 "{data: 1}"
ros2 topic pub --once /vision_mode std_msgs/msg/UInt8 "{data: 2}"
ros2 topic echo --qos-reliability best_effort /vision/status
ros2 service call /vision/get_status cv3_interfaces/srv/GetVisionStatus "{}"
```

实际硬件结果应单独记录，包括相机单次打开、模式切换、窗口、检测、缓存失效、
退出后重开和标定行为。

## 第三方来源

色环算法的来源、改动和 CC BY-SA 4.0 许可说明见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
