import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="svstools",
    version="0.2.1",
    author="Burak Kakillioglu",
    author_email="bkakilli@syr.edu",
    description="Smart Vision Systems Lab miscellaneous utility and visualization tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SYR-SVS-LAB/svstools",
    packages=['svstools'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "matplotlib",
        "tqdm",
        "requests",
        "open3d>=0.9"
    ]
)
