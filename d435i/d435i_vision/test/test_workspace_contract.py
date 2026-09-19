from pathlib import Path
import xml.etree.ElementTree as ET


GROUP_ROOT = Path(__file__).resolve().parents[2]
INTERFACES = GROUP_ROOT / "cv3_interfaces"


def compact(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_group_root_is_not_a_ros_package() -> None:
    assert not (GROUP_ROOT / "package.xml").exists()


def test_interface_package_name_and_build_type() -> None:
    package_xml = ET.parse(INTERFACES / "package.xml").getroot()
    assert package_xml.findtext("name") == "cv3_interfaces"
    assert package_xml.findtext("export/build_type") == "ament_cmake"


def test_interface_wire_contract_is_preserved() -> None:
    assert compact(INTERFACES / "srv" / "GetVisionStatus.srv") == [
        "---",
        "bool cache_available",
        "float64 result_age_s",
        "cv3_interfaces/VisionStatus status",
    ]
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
