from setuptools import setup, find_packages

setup(
    name="angular_resection",
    version="0.1.0",
    description="2D bearing-only resection and localization tools",
    author="Trung Pham",
    packages=find_packages("./src"),
    install_requires=[
        "numpy",
        "scipy",
        "pyproj",
        "folium",
        # add other dependencies as needed
    ],
    python_requires=">=3.8",
)