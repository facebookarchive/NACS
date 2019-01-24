# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
import sys
from torch.utils.ffi import create_extension


bleu = Extension(
    'seq2seq.libbleu',
    sources=[
        'seq2seq/clib/libbleu/libbleu.cpp',
        'seq2seq/clib/libbleu/module.cpp',
    ],
    # On Mac OS X, add the following to the extra_compile_args list:
    # '-mmacosx-version-min=10.11'
    extra_compile_args=['-std=c++11', '-mmacosx-version-min=10.11'],
)


if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')


with open('README.md') as f:
    readme = f.read()


with open('LICENSE') as f:
    license = f.read()


with open('requirements.txt') as f:
    reqs = f.read()


setup(
    name='NACS',
    version='0.1.0',
    description='NACS',
    long_description=readme,
    license=license,
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(),
    ext_modules=[bleu],
)
