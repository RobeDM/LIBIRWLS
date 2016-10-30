# -*- coding: UTF-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os

os.environ["CC"]=os.environ.get("CC","gcc")

print ""
print "USING COMPILER:",os.environ["CC"]
print ""

VecLibDir = os.environ.get("VECLIBDIR","/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/")

print ""
print "VECLIB ROUTINES:",VecLibDir
print ""

libgompPath=os.environ.get("LIBGOMP_PATH","libgomp.a")

print ""
print "USING LIBGOMP:",libgompPath
print ""

ext_modules = [ 
        Extension('LIBIRWLS',
            include_dirs=['.','../include/',np.get_include()],
            sources = ['pythonmodule.c'],
            extra_objects = [libgompPath,
                '../build/LIBIRWLS-predict.o',
                '../build/PIRWLS-train.o',
                '../build/PSIRWLS-train.o',
                '../build/IOStructures.o',
                '../build/ParallelAlgorithms.o',
                '../build/kernels.o'
            ],
            library_dirs = [VecLibDir,"../build/"],
            extra_compile_args=['-Wno-cpp','-static','-lgomp','-lblas','-llapack'],
            extra_link_args=['-static']
        )
        ]

setup(
        name = 'LIBIRWLS',
        version = '2.0',
        description="A Parallel IRWLS procedure for SVMs and Semiparametric SVMs",
        install_requires=[
            'numpy',
            'scipy',
        ],
        url='https://robedm.github.io/LIBIRWLS/',
        ext_modules = ext_modules
      )
