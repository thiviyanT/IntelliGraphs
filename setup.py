from setuptools import setup, find_packages

setup(
    name="intelligraphs",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'tqdm',
        'bokeh',
        'graphviz',
    ],
    author="Thiviyan Thanapalasingam",
    author_email="thiviyan.t@gmail.com",
    description="A simple library to generate synthetic Knowledge Graphs using random triples.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thiviyanT/intelligraphs",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires='>=3.7',
)

