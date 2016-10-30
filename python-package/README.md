# PYTHON EXTENSION

To use this module you must have the libraries numpy and Cython installed in python.

 - [Cython] (http://cython.org/)
 - [Numpy] (http://www.numpy.org/)

You can install very easily both libraries using the command [pip](https://pip.pypa.io/en/stable/).

## Installation Instructions:

### Requeriments:

You need to build the application before installing the python extension. To do that follow the instructions in the main README.md file.

### Windows

If you Currently this python extension is still not available for Windows operating systems.

### Installation Linux, Unix

If you have manually installed ATLAS (instead of using the apt-get command), you must tell the installation directory by defining the environment variable ATLASDIR:

    export ATLASDIR=/path/to/atlas/

After that, you can easily install this python module: 

    sudo python setup.py install
    
Remember: If you need sudo permission you have to use the parameter -E to keep the environment variable:

    sudo -E python setupOSX.py install
    
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

With these environment variables well defined you can install 

    python setupOSX.py install
    
Remember: If you need sudo permission you have to use the parameter -E to keep the environment variables:

    sudo -E python setupOSX.py install

## RUNNING:

From python, to test that this module was installed type:

        import LIBIRWLS

These are the available functions available of the module:

- To train SVM using the PIRWLS algorithm (data is a numpy 2d array with the training set features, labels is a numpy array with the training set labels, the other parameters and its possible values are detailed in the main README.md file in the command line instructions), the only mandatory parameters are data and labels:

        model = LIBIRWLS.PIRWLStrain(data, labels, gamma=1, C=1, threads=1, workingSet=500, eta=0.001, kernel=1)


- To train SVM using the PSIRWLS algorithm (data is a numpy 2d array with the training set features, labels is a numpy array with the training set labels, the other parameters and its possible values are detailed in the main README.md file in the command line instructions), the only mandatory parameters are data and labels::

        model = LIBIRWLS.PSIRWLStrain(data, labels, gamma=1, C=1, threads=1, size=500, algorithm=0.001, kernel=1)

- To classify a new dataset (the only mandatory parameters are model and data):

        predictions = LIBIRWLS.predict(model, data, labels, Threads=1, Soft=0)

- To save a model in a file:

        LIBIRWLS.save(model, filename)

- To save a model in a file:

        model = LIBIRWLS.load(filename)



