from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="EB_MLOPS_MID_PROJECT_1",
    version="0.1",
    author="EaditBernstein",
    packages=find_packages(),
    install_requires = requirements,
)