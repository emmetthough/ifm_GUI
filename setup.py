from setuptools import setup, find_packages

setup(
    name="ifm_GUI",  # Your project name
    version="0.1",  # Initial version
    packages=find_packages(),  # Automatically find your project packages
    install_requires=[  # Dependencies that are needed for your project
        "numpy>=1.21.0",
        "matplotlib>=3.4.0"
    ],
    author="Emmett Hough",
    author_email="emmett.hough@gmail.com",
    description="Developmental code for continuous acquiaition of interferometer absoption images and automatic population of fringes.",
    long_description=open("README.md").read(),  # Long description from README
    long_description_content_type="text/markdown",  # Format of long description
    url="https://github.com/emmetthough/ifm_GUI",  # Link to your GitHub project
    classifiers=[  # Some classifiers that help others find your project
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Specify Python version compatibility
)

