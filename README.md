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
        +-- IOStructures.c
        +-- LIBIRWLS-predict.c
        +-- PIRWLS-train.c
        +-- PSIRWLS-train.c
        +-- ParallelAlgorithms.c
        +-- kernels.c

Files and folders:
* **README.md**: This markdown file
* **Makefile**: The file with the directives used with the make build automation tool.
* **bin/**: It appears when the application is build using the make command and it contains the executable binaries.
* **build/**: It appears when the application is build using the make command and it contains the C object files.
* **docs/html/**: A detailed documentation of every function of source code in html format.
* **docs/latex/**: A detailed documentation of every function of source code in latex format (it includes a Makefile to build a pdf documentation).
* **examples/**: Folder with some script examples to run the algorithms.
* **include/**: Folder with the C headers used.
* **src/**: Folder with the C source code.

Online documentation:
=============

A documentation of the functions has been created in html format and it can be found in the folder docs/html. This documentation is also available online:

 - [Online Documentation] (http://robedm.github.io/LIBIRWLS/) Generated using doxygen.


Requirements:
=============

This software is implemented in C and requires the following libraries:

 - [OpenMP] (http://openmp.org/wp/) To parallelize the software
 - A Linear Algebra Package that implements the BLAS and Lapack standard routines, this software has been tested with these libraries:
     - [BLAS] (http://www.netlib.org/blas/), [LAPACK] (http://www.netlib.org/lapack/) and [ATLAS] (http://math-atlas.sourceforge.net/)
     - [MKL](https://software.intel.com/en-us/intel-mkl)


Installation Instructions:
=========

External libraries:
________________

An example to install in ubuntu all the libraries that we need is the following (apt-get is command in the debian based linux distributions to install packages):

OPENMP is currently included with the gcc compiler, if gcc is not installed, use the following command line:

    sudo apt-get install build-essential


To install the linear algebra routines use the following command line:

    sudo apt-get install liblapack-dev
    sudo apt-get install libblas-dev
    sudo apt-get install libatlas-dev

Compiling:
__________

If you use blas, lapack, atlas:

    make

If you use MKL libraries:

    make USE_MKL=1


If the libraries are not installed in the standard paths you can edit the file Makefile and uncomment and edit the following variables:

 **INCLUDEPATH** to tell the cblas.h file
 
 **LIBRARYPATH** to tell the linear algebra libraries location.


Running the code:
=================

Training:
________

To train the algorithm and create the model:

    ./PIRWLS-train [options] training_set_file model_file

training_set_file: Training set in LibSVM format
model_file: File where the classifier will be stored

Options:
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

    ./PIRWLS-predict [options] dataset_file model_file output_file

Options:
* -t Number_of_Threads: It is the number of threads in the parallel task (default 1)
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
