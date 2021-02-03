import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eeg-classification-threshold-jernejvivod",
    version="0.0.1",
    author="JernejVivod",
    author_email="vivod.jernej@gmail.com",
    description="EEG classification threshold demo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jernejvivod/eeg-classification-threshold",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)

