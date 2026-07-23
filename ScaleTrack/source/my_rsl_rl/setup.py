import os
import toml

from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "torch>=2.6.0",
    "torchvision>=0.5.0",
    "tensordict>=0.7.0",
    "numpy>=1.16.4",
    "GitPython",
    "onnx",
]

# Installation operation
setup(
    name="my_rsl_rl",
    packages=["my_rsl_rl"],
    author="Weishuai Zeng",
    url="https://github.com/zengweishuai/ScaleBFM",
    version="0.1.0",
    description="Self-modifed Version of RSL-RL",
    keywords="Scaling; Humanoid; RSL-RL",
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)
