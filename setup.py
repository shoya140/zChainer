# coding: utf-8

try:
    import setuptools
    from setuptools import setup, find_packages
except ImportError:
    print("Please install setuptools.")

import os
long_description = 'scikit-learn like interface and stacked autoencoder for chainer'
if os.path.exists('README.txt'):
    long_description = open('README.txt').read()

setup(
    name  = 'zChainer',
    version = '0.3.1',
    description = 'scikit-learn like interface and stacked autoencoder for chainer',
    long_description = long_description,
    license = 'MIT',
    author = 'Shoya Ishimaru',
    author_email = 'shoya.ishimaru@gmail.com',
    url = 'https://github.com/shoya140/zChainer',
    keywords = 'deep neural network, machine learning',
    packages = find_packages(),
    install_requires = ['numpy', 'chainer>=1.5', 'scikit-learn'],
    classifiers = [
      'Programming Language :: Python :: 2.7',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License'
    ]
)
