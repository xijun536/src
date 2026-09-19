from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_declares_runtime_dependencies() -> None:
    package = ET.parse(ROOT / "package.xml").getroot()
    assert package.findtext("name") == "d435i_vision"
    assert package.findtext("export/build_type") == "ament_python"
    dependencies = {node.text for node in package.findall("exec_depend")}
    assert {"rclpy", "cv3_interfaces", "ament_index_python"} <= dependencies


def test_setup_installs_console_script_config_and_launch() -> None:
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "d435i_vision = d435i_vision.main:main" in setup
    assert "config/config.yaml" in setup
    assert "launch/*.launch.py" in setup


def test_launch_exposes_config_and_headless_arguments() -> None:
    launch = (ROOT / "launch" / "d435i.launch.py").read_text(
        encoding="utf-8"
    )
    assert 'DeclareLaunchArgument("config_file"' in launch
    assert 'DeclareLaunchArgument("no_display"' in launch
    assert 'package="d435i_vision"' in launch
    assert 'executable="d435i_vision"' in launch
