from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="breakshell",
    version="0.2.0",
    author="Greatbeing",
    author_email="being19@163.com",
    description="BreakShell — AI Agent 自我模型安全层",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Greatbeing/minimal-agency",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
        "financial": ["pandas>=2.0"],
    },
    entry_points={
        "console_scripts": [
            "breakshell=breakshell.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
