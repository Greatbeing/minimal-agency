from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="breakshell",
    version="1.0.0",
    author="Greatbeing",
    author_email="being19@163.com",
    description="BreakShell — AI Agent Self-Model Safety Layer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Greatbeing/minimal-agency",
    packages=find_packages(where=".", exclude=["tests*", "tests"]),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "breakshell=breakshell_pkg.breakshell.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)