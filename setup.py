from setuptools import setup, find_packages

setup(
    name="garmin-activity-tracker",
    version="0.1.0",
    description="A Python desktop tool for analyzing and visualizing Garmin running activities with AI-powered insights.",
    author="Ryan Kari",
    author_email="ryan.j.kari@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "PyQt5>=5.15",
        "matplotlib",
        "pandas",
        "numpy",
        "Jinja2",
        "garminconnect",
        "openpyxl",
        "PyQt5",
        "ollama",
        "pytest",
        "jinja2",
    ],
    entry_points={
        "console_scripts": [
            "garmin-activity-tracker = main:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)