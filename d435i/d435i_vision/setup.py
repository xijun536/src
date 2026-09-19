from glob import glob
from setuptools import find_packages, setup


package_name = "d435i_vision"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/d435i_vision"],
        ),
        (
            "share/d435i_vision",
            ["package.xml", "README.md", "THIRD_PARTY_NOTICES.md"],
        ),
        ("share/d435i_vision/config", ["config/config.yaml"]),
        ("share/d435i_vision/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="book",
    maintainer_email="book@example.com",
    description="D435i surface tracking and ring calibration vision node.",
    license="CC-BY-SA-4.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "d435i_vision = d435i_vision.main:main",
        ],
    },
)
