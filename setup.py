# coding=utf-8

from setuptools import setup, find_packages

setup(
    name="torch_mgdcf",
    python_requires='>3.7.0',
    version="0.0.1",
    author="Jun Hu",
    author_email="hujunxianligong@gmail.com",
    packages=find_packages(
        exclude=[
            'clibs',
            'datasets',
            'run_mgdcf_yelp.sh',
            'run_mgdcf_gowalla.sh',
            'run_mgdcf_amazon-book.sh',
            'main_light_gcn.py',
            'main_mgdcf.py'
        ]
    ),
    install_requires=[
        "numpy >= 1.17.4",
        "requests",
        "tqdm",
        "scikit-learn >= 0.22",
        "faiss-cpu"
    ],
    extras_require={
    },
    package_data={'torch_mgdcf': ["metrics/libranking.so", "metrics/libranking.dll"]},
    description="MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering",
    license="GNU General Public License v3.0 (See LICENSE)",
    # long_description=open("README.rst", "r", encoding="utf-8").read(),
    long_description="MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering",
    url="https://github.com/CrawlScript/Torch-MGDCF"
)