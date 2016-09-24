# LIBIRWLS


LIBIRWLS is an integrated parallel library for Support Vector Machines (SVMs) that makes use of the IRWLS procedure. It implements the functions to run two different algorithms:

**Parallel Iterative Re-Weighted Least Squares:** A Parallel SVM solver based on the IRWLS algorithm.

**Parallel Semi-parametric Iterative Re-Weighted Least Squares:** A Parallel solver for semiparametric SVMs solver based on the IRWLS algorithm.

License: MIT License
====================

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

Directory List:
=============

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
    |   +-- demoIRWLS.bat
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
    +-- src/
    |   +-- IOStructures.c
    |   +-- LIBIRWLS-predict.c
    |   +-- PIRWLS-train.c
    |   +-- PSIRWLS-train.c
    |   +-- ParallelAlgorithms.c
    |   +-- kernels.c
    |
    +-- windows/
        +-- PIRWLS-train.exe
        +-- PSIRWLS-train.exe
        +-- LIBIRWLS-predict.exe

Files and folders:
* **README.md**: This markdown file
* **Makefile**: The file with the directives used with the make build automation tool.
* **bin/**: It appears when the application is build using the make command and it contains the executable binaries.
* **build/**: It appears when the application is build using the make command and it contains the C object files.
* **data/**: Some sample data needed to run the demo scripts.
* **demo/**: A .bat windows demo script and a Unix .sh demo script that runs the executable files.
* **docs/html/**: A detailed documentation of every function of source code in html format.
* **docs/latex/**: A detailed documentation of every function of source code in latex format (it includes a Makefile to build a pdf documentation).
* **examples/**: Folder with some script examples to run the algorithms.
* **include/**: Folder with the C headers used.
* **src/**: Folder with the C source code.
* **windows/**: Precompiled windows executable files.

Web:
=============

You can find detailed information about the software and the algorithm in its respective webpage:

 - [LIBIRWLS] (http://robedm.github.io/LIBIRWLS/index.html).


Online API:
=============

A documentation of the application programming interface (API) has been created in html format and it can be found in the folder docs/html. This documentation is also available online:

 - [Online Documentation] (http://robedm.github.io/LIBIRWLS/API/index.html) Generated using doxygen.


Installation Instructions:
=========

Requeriments:
____________

This software is implemented in C and requires the following libraries:

 - [OpenMP] (http://openmp.org/wp/) To parallelize the software.
 - [BLAS] (http://www.netlib.org/blas/) and [LAPACK] (http://www.netlib.org/lapack/): Linear algebra packages with standard routines.

Install:
________________

### Windows


LIBIRWLS contains windows static executable files that were precompiled for 64bits instancies. These executables are static so no extra packages are needed.


### Linux, Unix

#### Dependencies:

 - Ubuntu, Debian, Slackware and other linux distributions using the Advanced Package Tools:

    The Advanced Package Tool, or APT, is a free software user interface that works with core libraries to handle the installation and removal of software on some Linux distributions. If gcc is not installed, use the following command line:

        sudo apt-get install build-essential

    To install the linear algebra routines use the following command line:

        sudo apt-get install liblapack-dev
        sudo apt-get install libblas-dev

 - If you have any Linux or Unix distribution with no apt-get support you need to download BLAS and lapack from their official webpages and install them following their instructions. [In this link](https://pheiter.wordpress.com/2012/09/04/howto-installing-lapack-and-blas-on-mac-os/) you can find simple and detailed instructions to do that.

#### Compiling:

You need to run make in the library folder:

    cd LIBIRWLS
    make

### Mac OS X

#### Dependencies:

This software needs BLAS and Lapack standard routines. We have two different alternatives:

 - If you are a Mac user, there is no need to install them because Mac comes with BLAS and LAPACK since Maverick. The names of the files are libBLAS.dylib and libLAPACK.dylib, in some OS X versions they are in the folder:

        /System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Versions/Current/

    The folder may change depending on the OS X version, you can use the command find to find out the location:

        sudo find /System/Library/ -name "libBLAS.dylib"
        sudo find /System/Library/ -name "libLAPACK.dylib"

    These libraries are not in a library path, you need to put a copy of them in the folder /usr/local/lib

 - The alternative is to download both libraries, compile them and put the libraries in the library path.[In this link](https://pheiter.wordpress.com/2012/09/04/howto-installing-lapack-and-blas-on-mac-os/) you can find simple and detailed instructions to do that.

#### Compiling:

The default compiler installed in OS X is clang. It currently doesn't support openmp to perform the parallelization. We recommend the installation of gcc using Homebrew or Macports:

 - Macports (this option is faster): Download and Install macports from [https://www.macports.org/](https://www.macports.org/) and install gcc using the following command line:

    sudo port install gcc49

 - Homebrew: Install homebrew using the following command line:

        /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

    and then install gcc using the command line:

        brew install gcc --without-multilib


Then you can use the make command telling the path of the compiler that you have installed (the default path for macports gcc is /opt/local/bin/ and the default path for homebrew is /usr/local/Cellar/), for example:

 - If you have installed gcc 4.9 using Macports:

        cd LIBIRWLS
        make CC=/opt/local/bin/gcc-mp-4.9

 - If you have installed gcc 6 using Homebrew:

        cd LIBIRWLS
        make CC=/usr/local/Cellar/gcc/6.2.0/bin/gcc-6

        

Running the code:
=================

Training using the PSIRWLS algorithm:
________

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


Example:

    ./PSIRWLS-train -g 0.001 -c 1000 -t 4 -s 150 training_set_file.txt model_file.mod


Training using the PIRWLS algorithm:
________

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


Test:
_____

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

Input file format:
=================

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
