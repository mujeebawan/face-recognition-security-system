"""
Setup script for NRTC Face AI - Proprietary Face Recognition Library
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
version = {}
with open(os.path.join("nrtc_faceai", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

# Read README
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="nrtc-faceai",
    version=version.get("__version__", "1.0.0"),
    author="NRTC",
    author_email="info@nrtc.com.pk",
    description="NRTC Face AI - Proprietary Face Recognition Library for Jetson Devices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://nrtc.com.pk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "insightface>=0.7.0",
        "scikit-learn>=1.0.0",
        "onnxruntime-gpu>=1.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    license="Proprietary",
    keywords="face-recognition face-detection insightface arcface jetson cuda gpu ai nrtc",
)
