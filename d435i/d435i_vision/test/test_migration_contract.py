import ast
from pathlib import Path

from d435i_vision.resource_paths import resolve_default_config_path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = PACKAGE_ROOT / "d435i_vision"


def test_installed_share_path_points_to_installed_config() -> None:
    result = resolve_default_config_path(
        package_share_lookup=lambda _: "/tmp/fake-d435i-share"
    )
    assert result == Path("/tmp/fake-d435i-share/config/config.yaml")


def test_missing_ament_package_falls_back_to_new_source_root() -> None:
    def missing(_: str) -> str:
        raise LookupError("not installed")

    result = resolve_default_config_path(
        package_share_lookup=missing,
        source_root=PACKAGE_ROOT,
    )
    assert result == PACKAGE_ROOT / "config" / "config.yaml"


def test_runtime_sources_do_not_reference_old_project() -> None:
    for path in MODULE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "/home/book/Desktop/cv3" not in text
        tree = ast.parse(text, filename=str(path))
        imported_modules = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        imported_modules.extend(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        legacy_imports = [
            module
            for module in imported_modules
            if module == "cv3" or module.startswith("cv3.")
        ]
        assert legacy_imports == [], f"{path} imports legacy modules: {legacy_imports}"


def test_readme_documents_new_workspace_commands() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "colcon build --symlink-install" in readme
    assert "--packages-select cv3_interfaces d435i_vision" in readme
    assert "source /home/book/Desktop/install/setup.bash" in readme
    assert "ros2 launch d435i_vision d435i.launch.py" in readme
    assert "ros2 topic echo --qos-reliability best_effort /vision/status" in readme
    assert "ros2 service call /vision/get_status" in readme
