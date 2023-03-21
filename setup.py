import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="giz",
    version="0.1.0",
    author="Giz Team",
    author_email="mail@example.com",
    description="Solar panel calculator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aindrajaya/giz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "Pillow==8.4.0",
    ],
    entry_points={
        'console_scripts': [
            'my_crop_script=app.main:crop_image_script',
        ],
    },
)
