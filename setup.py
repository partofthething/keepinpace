from setuptools import setup, find_packages

with open("README.md") as f:
    README = f.read()

setup(
    name="keepinpace",
    version="0.1rc0",  
    description="A nuclear reactor kinetics code",
    author="Nick Touran",
    author_email="nick@partofthething.com",
    license="MIT",
    long_description=README,
    python_requires=">=3.6",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "ordered-set",
        "scipy",
    ],
)
