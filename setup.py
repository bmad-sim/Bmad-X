import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bmadx",
    version="0.0.1",
    author="J.P. Gonzalez-Aguilera",
    description="Experimental Bmad code transcribed in Python with Numba and Pytorch support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bmad-sim/Bmad-X",
    project_urls={"Bug Tracker": "https://github.com/bmad-sim/Bmad-X/issues"},
    license="Apache-2.0",
    packages=[
        package 
        for package in setuptools.find_packages(".")
        if package not in {"test"}
    ],
)
