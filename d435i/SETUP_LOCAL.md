# 本机运行（nuc / Ubuntu 24.04）

ROS 2 Humble 通过 RoboStack 安装在 Conda `d435i` 环境内，使用兼容的 Python 3.11 构建。
不要加载其他系统 ROS 的 setup.bash。旧 README 中 `/home/book` 路径不适用于本机。

连接 D435i 后直接启动：

```bash
/home/nuc/Desktop/src/d435i/run.sh
```

不显示图像窗口：

```bash
/home/nuc/Desktop/src/d435i/run.sh no_display:=true
```

手动加载环境：

```bash
conda activate d435i
source /home/nuc/Desktop/src/d435i/install/local_setup.bash
ros2 launch d435i_vision d435i.launch.py
```

重建（仅这两个包）：

```bash
conda activate d435i
cd /home/nuc/Desktop/src/d435i
colcon build --symlink-install \
  --base-paths cv3_interfaces d435i_vision \
  --packages-select cv3_interfaces d435i_vision \
  --cmake-args -G Ninja -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DPython_ROOT_DIR="$CONDA_PREFIX"
```

`environment.yml` 记录直接依赖；Empy 3.3.4 用于兼容 Humble 消息生成器，
NumPy 保持 1.x 以匹配 ROS 二进制接口，OpenCV 使用支持窗口的 Qt 构建。
相机 USB 权限规则安装在 `/etc/udev/rules.d/99-realsense-libusb.rules`。

依赖安装来源：https://robostack.github.io/conda.html

本次软件验证：项目 261 项测试全部通过；ROS 自定义消息可创建并发布，
节点 `--help` 与 launch `--show-args` 可用。随后连接 D435i，已验证 `simple_vision` 两个入口的真实彩色帧采集与窗口显示。
ROS 主节点的完整硬件运行尚未验收。

## 简单二维检测（无需 ROS）

```bash
conda activate d435i
cd /home/nuc/Desktop/src/d435i/simple_vision
python block_detector.py --color green
# 关闭上一个程序后运行：
python ring_detector.py
```

按 q 或 Esc 退出；两个程序不能同时占用相机。
使用 `--no-display --frames 3` 可进行无窗口采集检查。
`environment.resolved.yml` 是本机已安装版本快照，跨机器安装优先使用 `environment.yml`。
