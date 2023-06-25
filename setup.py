import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dpconvcnp",
    version="0.0.1",
    author="Stratis Markou, Ossi Raisa",
    author_email="em626@cam.ac.uk",
    description="Differentially Private Convolutional Conditional Neural Processes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)