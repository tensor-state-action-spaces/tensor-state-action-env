#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs 
import os
import sys
from shutil import rmtree

from setuptools import Command, find_packages, setup

# Package meta-data.
NAME = 'tensor_state_action_env'
DESCRIPTION = 'Environments for tensor based state and action environments'
URL = 'https://github.com/zhuyifengzju/tensor-state-action-env'
EMAIL = 'yifeng_zhu@outlook.com'
AUTHOR = 'Yifeng Zhu'

# What packages are required for this module to be executed?
REQUIRED = [
    'attr', 'numpy','gym', 'pygame', 'pillow'
]
# REQUIRED = [
#     'tensorflow', 'numpy', 'keras', 'attrs', 'dm-sonnet', 'semver', 'dill',
#     'pillow', 'gym', 'sacred', 'tqdm'
# ]

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.rst' is present in your
# MANIFEST.in file!
with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)

VERSION = about['__version__']

#  ____        _     _
# / ___| _ __ | |__ (_)_ __ __  __
# \___ \| '_ \| '_ \| | '_ \\ \/ /
#  ___) | |_) | | | | | | | |>  <
# |____/| .__/|_| |_|_|_| |_/_/\_\
#       |_|
#

try:
    from sphinx.setup_command import BuildDoc
    sphinx_cmdclass = {'build_sphinx': BuildDoc}
    sphinx_command_options = {
        'build_sphinx': {
            'project': ('setup.py', NAME),
            'version': ('setup.py', VERSION),
            'release': ('setup.py', VERSION),
            'build_dir': ('setup.py', os.path.join(here, 'docs', 'build')),
            'config_dir': ('setup.py', os.path.join(here, 'docs', 'source')),
        }
    }
except ImportError:
    sphinx_cmdclass = {}
    sphinx_command_options = {}

cmdclass = sphinx_cmdclass
command_options = sphinx_command_options

# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=('tests', )),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    include_package_data=True,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    # $ setup.py publish support.
    cmdclass=cmdclass,
    command_options=command_options)
