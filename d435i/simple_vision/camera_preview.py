"""RealSense 彩色相机实时检测入口，不依赖 ROS。"""

import argparse
import json

import cv2
import numpy as np
import pyrealsense2 as rs


def main(default_mode="blocks"):
    parser = argparse.ArgumentParser(description="D435i 实时二维检测，按 q 或 Esc 退出")
    parser.add_argument("--mode", choices=("blocks", "rings"), default=default_mode)
    parser.add_argument("--color", choices=("red", "green", "blue", "yellow", "black", "light_blue"), default="green")
    parser.add_argument("--no-display", action="store_true", help="仅采集并输出结果")
    parser.add_argument("--frames", type=int, default=0, help="采集帧数，0 表示持续运行")
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("--frames 必须大于等于 0")

    # 延迟导入，允许直接执行两个算法文件，也允许作为包导入。
    if __package__:
        from .block_detector import detect_blocks
        from .ring_detector import detect_rings
    else:
        from block_detector import detect_blocks
        from ring_detector import detect_rings

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    started = False
    title = "D435i - " + args.mode
    try:
        pipeline.start(config)
        started = True
        print("相机已启动，按 q 或 Esc 退出。", flush=True)
        frame_count = 0
        while args.frames == 0 or frame_count < args.frames:
            frame = pipeline.wait_for_frames(10000).get_color_frame()
            if not frame:
                continue
            image = np.asanyarray(frame.get_data()).copy()
            results = detect_blocks(image, color=args.color) if args.mode == "blocks" else detect_rings(image)
            for result in results:
                center = tuple(round(v) for v in result["center_px"])
                if args.mode == "blocks":
                    cv2.circle(image, center, round(result["radius_px"]), (0, 255, 0), 2)
                else:
                    axes = (round(result["major_axis_px"] / 2), round(result["minor_axis_px"] / 2))
                    cv2.ellipse(image, center, axes, result["angle_deg"], 0, 360, (0, 255, 0), 2)
                cv2.circle(image, center, 3, (0, 0, 255), -1)
            cv2.putText(image, f"{args.mode}: {len(results)}  color: {args.color}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            frame_count += 1
            if frame_count == 1 or frame_count % 30 == 0:
                print(json.dumps({"frame": frame_count, "detections": results}), flush=True)
            if not args.no_display:
                cv2.imshow(title, image)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if started:
            pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
