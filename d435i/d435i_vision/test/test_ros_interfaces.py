from __future__ import annotations

from pathlib import Path


GROUP_ROOT = Path(__file__).resolve().parents[2]
INTERFACES = GROUP_ROOT / "cv3_interfaces"


def compact(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_ring_center_message_contract() -> None:
    assert compact(INTERFACES / "msg" / "RingCenter.msg") == [
        "uint8 order",
        "float64 center_u_px",
        "float64 center_v_px",
        "bool position_3d_valid",
        "geometry_msgs/Point position_camera_m",
        "float64 major_axis_px",
        "float64 minor_axis_px",
        "float64 ellipse_angle_deg",
        "float64 fit_error",
    ]


def test_surface_tracking_message_contract() -> None:
    assert compact(INTERFACES / "msg" / "SurfaceTrackingResult.msg") == [
        "bool active",
        "bool tracking_valid",
        "bool measured",
        "string color",
        "float64 center_u_px",
        "float64 center_v_px",
        "float64 velocity_u_px_s",
        "float64 velocity_v_px_s",
        "float64 speed_px_s",
        "float64 radius_px",
        "uint32 missing_frames",
        "bool depth_applied",
        "bool surface_3d_valid",
        "geometry_msgs/Point surface_center_camera_m",
        "geometry_msgs/Vector3 surface_normal_camera",
        "float64 surface_inlier_ratio",
        "float64 normal_to_gravity_deg",
    ]


def test_ring_calibration_message_contract() -> None:
    assert compact(INTERFACES / "msg" / "RingCalibrationResult.msg") == [
        "bool active",
        "uint32 ellipse_candidate_count",
        "uint32 ring_count",
        "uint32 valid_3d_count",
        "cv3_interfaces/RingCenter[] rings",
        "bool exactly_three",
        "bool all_depth_valid",
        "bool collinear",
        "float64 collinearity_error_px",
        "float64 middle_error_u_px",
        "float64 middle_error_v_px",
        "float64 middle_center_error_px",
        "geometry_msgs/Point middle_center_camera_m",
        "geometry_msgs/Vector3 middle_error_camera_m",
        "float64 target_distance_m",
        "float64 distance_error_m",
        "geometry_msgs/Vector3 line_direction_camera",
        "float64 line_to_x_axis_angle_deg",
        "bool center_aligned",
        "bool distance_aligned",
        "bool angle_aligned",
        "bool calibration_ready",
        "string failure_reason",
    ]


def test_vision_status_and_service_contract() -> None:
    assert compact(INTERFACES / "msg" / "VisionStatus.msg") == [
        "std_msgs/Header header",
        "uint64 frame_sequence",
        "uint8 current_mode",
        "string current_mode_name",
        "string selected_color",
        "bool camera_ok",
        "bool result_available",
        "float64 processing_fps",
        "cv3_interfaces/SurfaceTrackingResult surface",
        "cv3_interfaces/RingCalibrationResult ring",
    ]
    assert compact(INTERFACES / "srv" / "GetVisionStatus.srv") == [
        "---",
        "bool cache_available",
        "float64 result_age_s",
        "cv3_interfaces/VisionStatus status",
    ]


def test_interface_package_generates_every_contract() -> None:
    cmake = (INTERFACES / "CMakeLists.txt").read_text(encoding="utf-8")
    for name in (
        "RingCenter.msg",
        "SurfaceTrackingResult.msg",
        "RingCalibrationResult.msg",
        "VisionStatus.msg",
        "GetVisionStatus.srv",
    ):
        assert name in cmake
    assert "DEPENDENCIES std_msgs geometry_msgs" in cmake

    package = (INTERFACES / "package.xml").read_text(encoding="utf-8")
    assert 'email="book@example.com"' in package
    assert "rosidl_default_generators" in package
    assert "rosidl_default_runtime" in package
    assert "rosidl_interface_packages" in package
