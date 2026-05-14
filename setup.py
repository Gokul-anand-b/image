from setuptools import setup

setup(
    name="gokul_image_lib",
    version="1.0",
    py_modules=["gokul_image_lib"],
    install_requires=[
        "opencv-python",
        "numpy",
        "matplotlib",
        "ultralytics"
    ],
)
