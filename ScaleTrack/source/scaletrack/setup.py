import os
import toml

from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "psutil",
    "onnxscript",
    "wandb>=0.19",
    "rich",
    "joblib",
    "ipdb",
    "easydict",
    "torch_tensorrt==2.8.0",
]

# Installation operation
setup(
    name="scaletrack",
    packages=["scaletrack"],
    author="Weishuai Zeng",
    url="https://github.com/zengweishuai/ScaleBFM",
    version="0.1.0",
    description="Scaling Behavior Foundation Model for Humanoid Robots",
    keywords="Scaling; Humanoid; Behavior Foundation Model",
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.11",
    zip_safe=False,
)
