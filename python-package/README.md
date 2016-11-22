# PYTHON EXTENSION

To use this module you must have the libraries numpy and Cython installed in python.

 - [Cython] (http://cython.org/)
 - [Numpy] (http://www.numpy.org/)

You can install very easily both libraries using the command [pip](https://pip.pypa.io/en/stable/).

## Installation Instructions:

### Requeriments:

You need to build the application before installing the python extension. To do that follow the instructions in the main README.md file.

### Windows

Currently this python extension is still not available for Windows operating systems.

### Installation Linux, Unix

If you have manually installed ATLAS (instead of using the apt-get command), you must tell the installation directory by defining the environment variable ATLASDIR:

    export ATLASDIR=/path/to/atlas/

After that, you can easily install this python module (If you need sudo permission you have to use the parameter -E to keep the environment variable): 

    sudo -E python setup.py install
    
### OS X

You must define some enviroment variables:

CC to tell the compiler (use the same compiler that you use to compile the command line app). For example, in the case of gcc 6 installed with homebrew:

    export CC=/usr/local/Cellar/gcc/6.2.0/bin/gcc-6
    
Only if veclib is not in the default directory you must define the environment variable VECLIBDIR telling where veclib is:

    export VECLIBDIR=/path/to/veclib

Your compiler has the openmp functions in the libgomp library, you must define an enviroment variable called LIBGOMP_PATH telling where is the file libgomp.a in the lib directory of your compiler. In the case of gcc 6 installed with homebrew: 

    export LIBGOMP_PATH=/usr/local/Cellar/gcc/6.2.0/lib/gcc/6/libgomp.a
    
To avoid that Cython could use any posible flag only available for clang it is better to set the CFLAGS environment variable to an empty value:

    export CFLAGS=

With these environment variables well defined you can install the extension (If you need sudo permission you have to use the parameter -E to keep the environment variables):

    sudo -E python setupOSX.py install

## RUNNING:

### Import this library:

        import LIBIRWLS


### PIRWLS algorithm:

It trains a SVM using a parallel IRWLS procedure. See the library [webpage](https://robedm.github.io/LIBIRWLS/) for a detailed description.


    model = LIBIRWLS.PIRWLStrain(data, labels, gamma=1, C=1, threads=1, workingSet=500, eta=0.001, kernel=1)

Parameters:
* data: Training set (numpy 2d array)
* labels: Training set labels (the label of every training data numpy array with values +1 and -1)
* kernel: kernel type: 
    * 0 for Linear kernel u'*v
    * 1 for radial basis function exp(-gamma*|u-v|^2)
* gamma: gamma in the radial basis kernel function
* C: SVM Cost
* workingSet: Size of the Least Squares Problem in every iteration
* threads: It is the number of parallel threads
* eta: Stop criteria

### PSIRWLS algorithm:

It trains a semiparametric SVM using a parallel IRWLS procedure. See the library [webpage](https://robedm.github.io/LIBIRWLS/) for a detailed description.

        model = LIBIRWLS.PSIRWLStrain(data, labels, gamma=1, C=1, threads=1, size=500, algorithm=0.001, kernel=1)

Parameters:
* data: Training set (numpy 2d array)
* labels: Training set labels (the label of every training data numpy array with values +1 and -1)
* kernel: kernel type: 
    * 0 for Linear kernel u'*v
    * 1 for radial basis function exp(-gamma*|u-v|^2)
* gamma: gamma in the radial basis kernel function
* C: SVM Cost
* threads: It is the number of parallel threads
* size: Size of the classifier
* algorithm: Algorithm for centroids selection
     * 0 -- Random Selection
     * 1 -- SGMA (Sparse Greedy Matrix Approximation

### To classify a new dataset:

        predictions = LIBIRWLS.predict(model, data, labels=None, Threads=1, Soft=0)

Parameters:
* model: A model obtained using PIRWLS or PSIRWLS.
* data: A dataset to classify.
* labels: dataset labels (optional)
* Threads: It is the number of parallel threads
* Soft:
    * 0 Class prediction (the output is +1 or -1)
    * 1 Soft output: The output after the hard decision that decides the class (useful to use in ensembles with other algorithms).


### To save a model in a file:

        LIBIRWLS.save(model, filename)

### To save a model in a file:

        model = LIBIRWLS.load(filename)



