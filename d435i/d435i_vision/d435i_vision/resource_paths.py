from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional


PackageShareLookup = Callable[[str], str]


def resolve_default_config_path(
    package_share_lookup: Optional[PackageShareLookup] = None,
    source_root: Optional[Path] = None,
) -> Path:
    """Locate installed config, falling back to this new source package."""

    lookup = package_share_lookup
    if lookup is None:
        try:
            from ament_index_python.packages import get_package_share_directory
        except ImportError:
            get_package_share_directory = None
        lookup = get_package_share_directory

    if lookup is not None:
        try:
            share = Path(lookup("d435i_vision"))
            return share / "config" / "config.yaml"
        except LookupError:
            pass

    root = source_root or Path(__file__).resolve().parents[1]
    return Path(root) / "config" / "config.yaml"
