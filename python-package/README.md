# PYTHON EXTENSION

## Installation Instructions:

### Requeriments:

You need to build the application before installing the python extension. To do that follow the instructions in the main README.md file.

### Windows

Currently this python extension is still not available for Windows operating systems.

### Installation Linux, Unix, OS X

If you default C compiler is gcc amd you have ATLAS installed in the library path you can install this module easily: 

        sudo python setup.py install

If your default gcc compiler is different (for example, if you had clang and you had to install gcc when building the library in OS X systems) you need to tell where is gcc:

        sudo python setup.py install

If ATLAS is not built in the library path you must edit the file setup.py to include the directories of the header files and libraries:

 - Include the lib folder under you ATLAS build directory in the list library directories:
 
        library_dirs = ['/path/to/atlas/lib/',"../build/"],

 - Include the include folder under you ATLAS build directory in the :
 
        include_dirs=['/path/to/atlas/include/','.','../include/',np.get_include()],

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



