import os
from setuptools import setup, find_packages


def read():
    return open(os.path.join(os.path.dirname(__file__), "README.md")).read()


setup(
    name="barazmoon",
    version="0.0.1",
    keywords=["load testing", "web service", "restful api"],
    packages=find_packages("."),
    long_description=read(),
    install_requires=[
        "numpy>=1.19.2",
        "aiohttp>=3.7.4",
    ]
)
