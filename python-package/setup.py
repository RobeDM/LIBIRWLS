# -*- coding: UTF-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os


AtlasDir=os.environ.get("ATLASDIR","..")

AtlasDir = AtlasDir+"/lib/"

ext_modules = [ 
        Extension('LIBIRWLS',
            include_dirs=['.','../include/',np.get_include()],
            sources = ['pythonmodule.c'],
            extra_objects = ['../build/LIBIRWLS-predict.o',
                '../build/PIRWLS-train.o',
                '../build/PSIRWLS-train.o',
                '../build/IOStructures.o',
                '../build/ParallelAlgorithms.o',
                '../build/kernels.o'
            ],
            library_dirs = [AtlasDir,"../build/"],
            extra_compile_args = ["-fPIC","-O3","-llapack", "-lf77blas", "-lcblas", "-latlas", "-lgfortran",'-fopenmp'],
            extra_link_args=["-fPIC","-llapack", "-lf77blas", "-lcblas", "-latlas", "-lgfortran",'-fopenmp']
        )
        ]

setup(
        name = 'LIBIRWLS',
        version = '2.1',
        description="A Parallel IRWLS procedure for SVMs and Semiparametric SVMs",
        install_requires=[
            'numpy',
            'scipy',
        ],
        url='https://robedm.github.io/LIBIRWLS/',
        ext_modules = ext_modules
      )
