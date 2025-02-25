from setuptools import setup, find_packages

setup(
    name="intelligraphs",
    version="1.0.17",
    packages=find_packages(),
    install_requires=[
        'tqdm',
        'bokeh',
        'graphviz',
        'requests',
        'torch',
        'seaborn',
        'matplotlib',
        'scipy',
        'numpy==1.26.4',
    ],
    author="Thiviyan Thanapalasingam",
    author_email="thiviyan.t@gmail.com",
    description="A Python package for using IntelliGraphs benchmarking datasets.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thiviyanT/intelligraphs",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires='>=3.7',
)

