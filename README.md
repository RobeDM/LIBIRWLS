# LIBIRWLS

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/8e33396866a9466f9c131c913c62b078)](https://www.codacy.com/app/rober-diaz/LIBIRWLS?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=RobeDM/LIBIRWLS&amp;utm_campaign=Badge_Grade)
[![Build Status](https://travis-ci.org/RobeDM/LIBIRWLS.svg?branch=master)](https://travis-ci.org/RobeDM/LIBIRWLS)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gitter](https://img.shields.io/gitter/room/badges/shields.svg)](https://gitter.im/LIBIRWLS/Lobby)

[WebPage](https://robedm.github.io/LIBIRWLS/) |
[API](https://robedm.github.io/LIBIRWLS/API/index.html) |
[Files](#directory-list) |
[Installation](#installation-instructions) |
[Running](#running-the-software-from-command-line)

## Description

LIBIRWLS is an integrated parallel library for Support Vector Machines (SVMs) that makes use of the IRWLS procedure. It implements the functions to run two different algorithms:

**Parallel Iterative Re-Weighted Least Squares:** A Parallel SVM solver based on the IRWLS algorithm.

**Parallel Semi-parametric Iterative Re-Weighted Least Squares:** A Parallel solver for semiparametric SVMs solver based on the IRWLS algorithm.

For a detailed explanation of the algorithms take a look at the [web page](https://robedm.github.io/LIBIRWLS/)

SVMs are a very popular machine learning technique because they can easily create non-linear solutions by transforming the input space onto a high dimensional one where a kernel function can compute the inner product of a pair vectors. Thanks to this ability, they offer a good compromise between complexity and performance in many applications.

![Dimensions](https://qph.ec.quoracdn.net/main-qimg-08fe68adf9ee3e05ca806e72cbd88b54?convert_to_webp=true)

## Motivation

SVMs have two main limitations. The first problem is related to their non-parametric nature. The complexity of the classifier is not limited and depends on the number of Support Vectors (SVs) after training. If the number of SVs is very large we may obtain a very slow classifier when processing new samples. The second problem is the run time associated to the training procedure that may be excessive for large datasets.

To face these problems, we can make use of parallel computing, thus reducing the run time of the training procedure or we can use semi-parametric approximations than can limit the complexity of the model in advance, which directly implies a faster classifier.

The above situation motivated us to develop "LIBIRWLS", an integrated library based on a parallel implementation of the IRWLS procedure to solve non-linear SVMs and semi-parametric SVMs. This library is implemented in C, supports a wide range of platforms and also provides detailed information about its programming interface and dependencies.

## License: MIT License

Copyright (c) 2015-2016 Roberto Diaz Morales, Ángel Navia Vázquez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## Directory List:

The structure of this library is as follows:


    LIBIRWLS/
    |
    +-- README.md
    |
    +-- Makefile
    |
    +-- bin/
    |
    +-- build/
    |
    +-- data/
    |
    +-- demo/
    |   +-- demoIRWLS.sh
    |   +-- demoIRWLSWin32.bat
    |   +-- demoIRWLSWin64.bat
    |
    +-- docs/
    |   |
    |   +— html/
    |   |
    |   +— latex/
    |
    +-- examples/
    |
    +-- inclue/
    |   +-- IOStructures.h
    |   +-- LIBIRWLS-predict.h
    |   +-- PIRWLS-train.h
    |   +-- PSIRWLS-train.h
    |   +-- ParallelAlgorithms.h
    |   +-- kernels.h
    |
    +-- python-package/
    |   +--pythonmodule.c
    |   +--pythonmodule.h
    |   +--setup.py
    |   +--setupOSX.py
    |
    +-- src/
    |   +-- IOStructures.c
    |   +-- LIBIRWLS-predict.c
    |   +-- PIRWLS-train.c
    |   +-- PSIRWLS-train.c
    |   +-- ParallelAlgorithms.c
    |   +-- kernels.c
    |
    +-- windows/
        |
        +--Win32
        |  +-- PIRWLS-train.exe
        |  +-- PSIRWLS-train.exe
        |  +-- LIBIRWLS-predict.exe
        |
        +--Win64
           +-- PIRWLS-train.exe
           +-- PSIRWLS-train.exe
           +-- LIBIRWLS-predict.exe


Files and folders:
* **README.md**: This markdown file
* **Makefile**: The file with the directives used with the make build automation tool.
* **bin/**: It appears when the application is build using the make command and it contains the executable binaries.
* **build/**: It appears when the application is build using the make command and it contains the C object files.
* **data/**: It appears when the demo scripts of the folder demo are executed and contains some sample data needed to run the demo scripts.
* **demo/**: A .bat windows demo script and a Unix .sh demo script that runs the executable files.
* **docs/html/**: A detailed documentation of every function of source code in html format.
* **docs/latex/**: A detailed documentation of every function of source code in latex format (it includes a Makefile to build a pdf documentation).
* **examples/**: Folder with some script examples to run the algorithms.
* **include/**: Folder with the C headers used.
* **python-module/**: Python extension to use this library.
* **src/**: Folder with the C source code.
* **windows/**: Precompiled windows executable files for 32 and 64 bits versions.

## Web:

You can find detailed information about the software and the algorithm in its respective webpage:

 - [LIBIRWLS] (http://robedm.github.io/LIBIRWLS/index.html).


## Online API:

A documentation of the application programming interface (API) has been created in html format and it can be found in the folder docs/html. This documentation is also available online:

 - [Online Documentation] (http://robedm.github.io/LIBIRWLS/API/index.html) Generated using doxygen.


## Installation Instructions:

### Requeriments:

This software is implemented in C and requires the following libraries:

 - [OpenMP] (http://openmp.org/wp/) To parallelize the software.
 - [ATLAS] (http://math-atlas.sourceforge.net/): Linear algebra package with standard routines that contains optimized BLAS and LAPACK implementations.

### Linux, Unix

#### Dependencies:

 - Ubuntu, Debian, Slackware and other linux distributions using the Advanced Package Tools:

    The Advanced Package Tool, or APT, is a free software user interface that works with core libraries to handle the installation and removal of software on some Linux distributions. If gcc is not installed, use the following command line:

        sudo apt-get install build-essential

    To install the linear algebra routines of ATLAS use the following command line:

        sudo apt-get install libatlas-base-dev

 - If you have any Linux or Unix distribution with no apt-get support you need to download ATLAS from the [official repository] (https://sourceforge.net/projects/math-atlas/files/) and install it following the instructions that are detailed in the file INSTALL.txt. If you are impatient, for a basic installation on a 64 bits computer, this is the basic outline:

        bzip2 -d atlas3.10.2.tar.bz2
        tar -xvf atlas3.10.2.tar.bz2
        cd ATLAS
        mkdir my_build_dir
        cd my_build_dir
        ../configure -b 64 --prefix=/installation/directory ! Tell the installation directory
        make                                                ! tune and compile library
        make check                                          ! perform sanity tests
        make ptcheck                                        ! checks of threaded code for multiprocessor systems
        make time                                           ! provide performance summary as % of clock rate
        make install                                        ! copy the library in the installation directory

#### Compiling:

You need to run make in the library folder to build LIBIRWLS. If you have installed atlas using apt-get:

    cd LIBIRWLS
    make

If you have manually installed ATLAS, you must tell the installation directory.

    cd LIBIRWLS
    make ATLASDIR=/installation/directory 


### Mac OS X

#### Compiler:

The default compiler installed in OS X is clang. It currently doesn't have a good support for openmp. We recommend the installation of gcc using [Homebrew](http://brew.sh/) or [Macports](https://www.macports.org/):

 - Homebrew: Install homebrew using the following command line:

        /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

    and then install gcc using the command line:

        brew install gcc --without-multilib

 - Macports: Download and Install macports from [https://www.macports.org/](https://www.macports.org/) and install gcc using the following command line:

        sudo port install gcc49


#### Dependencies:

OS X has its own accelerated algebra standard routines. The name of this library is veclib and it is composed by two files:

        libBLAS.dylib
        libLAPACK.dylib
    
If these files are commonly in the directory:

        /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/

Plese, check that both files are in the directory. If you cannot find them there, look for them using the command "find" and note the folder for the next step:

        sudo find / -name "libBLAS.dylib" 


#### Compiling:

You must use the make command using the following parameters:

 - OSX: A boolean variable that tells that you are using OS X operating system. 
 - CC: To tell where is the gcc compiler that you have installed.
 - VECLIBDIR: To tell where is the veclib library if it is not in the default directory.

Then you can use the make telling command telling the path of the compiler that you have installed (the default path for macports gcc is /opt/local/bin/ and the default path for homebrew is /usr/local/Cellar/) and the installation directory of ATLAS, for example:

 - For example, if you have installed gcc 6 using Homebrew:

        cd LIBIRWLS
        make OSX=1 CC=/usr/local/Cellar/gcc/6.2.0/bin/gcc-6

 - If you have installed gcc 6 using Homebrew and veclib is in a different directory called /veclib/directory then:

        cd LIBIRWLS
        make OSX=1 CC=/usr/local/Cellar/gcc/6.2.0/bin/gcc-6 VECLIBDIR=/veclib/directory

### Windows


LIBIRWLS contains windows executable files that were precompiled for 32 and 64 bits instancies. These executables are static so no extra packages are needed.

If you want to obtain an optimized performance the software must be compiled and built in your system using tools like [cygwin](https://www.cygwin.com/). This is because ATLAS fixes some parameters to optimize the run time attending to the microprocessor in the computer that builds it.


## Running the software from command line:

### Demo scripts:

For testing purposes, the folder demo contains a .bat windows demo script and a Unix .sh demo script that download a sample dataset from the libsvm repository and runs the executable files.


#### Training using the PSIRWLS algorithm:

The algorithm is described in this paper:

Díaz-Morales, R., & Navia-Vázquez, Á. (2016). Efficient parallel implementation of kernel methods. Neurocomputing, 191, 175-186.

To train the algorithm and create the model:

    ./PSIRWLS-train [options] training_set_file model_file

training_set_file: Training set in LibSVM format
model_file: File where the classifier will be stored

Options:
* -k kernel type: 0 = Linear kernel u'*v and 1 = radial basis function exp(-gamma*|u-v|^2) (default 1)
* -g Gamma: Set gamma in the radial basis kernel function (default 1)
* -c Cost: Set the SVM Cost (default 1)
* -s Classifier_size: Size of the classifier (default 50)
* -t Number_of_Threads: It is the number of threads in the parallel task (default 1)
* -a Algorithm: Algorithm for centroids selection (default 1)
     * 0 -- Random Selection
     * 1 -- SGMA (Sparse Greedy Matrix Approximation

Example:

    ./PSIRWLS-train -g 0.001 -c 1000 -t 4 -s 150 training_set_file.txt model_file.mod


#### Training using the PIRWLS algorithm:

The algorithm is described in this paper:

Morales, R. D., & Vázquez, Á. N. (2016). Improving the efficiency of IRWLS SVMs using Parallel Cholesky Factorization. Pattern Recognition Letters.

To train the algorithm and create the model:

    ./PIRWLS-train [options] training_set_file model_file

training_set_file: Training set in LibSVM format
model_file: File where the classifier will be stored

Options:
* -k kernel type: 
    * 0 for Linear kernel u'*v
    * 1 for radial basis function exp(-gamma*|u-v|^2) (default 1)
* -g Gamma: Set gamma in the radial basis kernel function (default 1)
* -c Cost: Set the SVM Cost (default 1)
* -w Working_set_size: Size of the Least Squares Problem in every iteration (default 500)
* -t Number_of_Threads: It is the number of threads in the parallel task (default 1)
* -e eta: Stop criteria (default 0.001)

Example:

    ./PIRWLS-train -g 0.001 -c 1000 -t 4 training_set_file.txt model_file.mod


#### Test:

To make predictions with the model in a different dataset:

    ./LIBIRWLS-predict [options] dataset_file model_file output_file

Options:
* -t Number_of_Threads: It is the number of threads in the parallel task (default 1)
* -s Soft output (default 0):
    * 0 Class prediction (the output is +1 or -1)
    * 1 Soft output: The output after the hard decision that decides the class (useful to use in ensembles with other algorithms).
* -l Labeled:  (default 0)
    * 1 if the dataset is labeled (shows accuracy)
    * 0 if the dataset is unlabeled

Example:

    ./PIRWLS-predict -t 4 -l 1 dataset_file.txt model_file.mod output_file.txt

### Input file format:

The dataset must be provided in LibSVM format, labeled to train the model and labeled or unlabeled for predictions (using the -l option in the PIRWLS-predict command to tell if the file is labeled or unlabeled):


Labeled example:

~~~~
+1 1:5 7:2 15:6
+1 1:5 7:2 15:6 23:1
-1 2:4 3:2 10:6 11:4
~~~~

Unlabeled example:

~~~~
1:5 7:2 15:6
1:5 7:2 15:6 23:1
2:4 3:2 10:6 11:4
~~~~

## Python module:

Installation and running instructions are detailed in the README.md file allocated in the folder python-module


