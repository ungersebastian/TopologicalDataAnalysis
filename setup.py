#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import setuptools
from __TDA_META__ import __author__, __author_email__, __description__, __License__, __Operating_system__, __Programming_language__,__title__, __url__, __version__

# If you want to incorporate c-files
# For the structure of an c-file -> Check resource files
#from distutils.core import  Extension
# define the extension module
# cos_module = Extension('cos_module', sources=['External_code/cos_module.c'])

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    # ext_mdoules = [External_code/cos_module],        <- for incorporating c-files
    name=__title__,
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['numpy>=1.16', 'sklearn' ],  
    include_package_data=True,
    package_data={'TopologicalDataAnalysis':['lens_functions/*']},
    packages=setuptools.find_packages(),
    classifiers=[
        __Programming_language__,
        __License__,
        __Operating_system__,
    ],
)
