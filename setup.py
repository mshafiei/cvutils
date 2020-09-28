import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cvgutilits",
    version="0.0.1",
    author="Mohammad Shafiei",
    author_email="shafieirn@gmail.com",
    description="Computer vision/graphics tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mshafiei/cvgutils.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)