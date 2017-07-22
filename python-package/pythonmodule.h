/*
 ============================================================================
 Author      : Roberto Diaz Morales
 ============================================================================
 
 Copyright (c) 2016 Roberto Díaz Morales

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
 (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge,
 publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
 ============================================================================
 */

/**
 * @brief Functions of the python extension that makes use of this library.
 *
 * A definition of the functions of this python extension
 * 
 * @file pythonmodule.h
 * @author Roberto Diaz Morales
 * @date 23 Sep 2016
 */ 


/**
 * @brief Model capsule destructor
 *
 * The capsule destructor of the model object
 *
 * @param pymodel The model to delete
 */
void delModel(PyObject* pymodel);


/**
 * @brief Python extension to classify data using a trained model on a numpy dataset.
 *
 * This is the function of the python extension to classify a numpy dataset using a trained model.
 *
 * @param dummy First parameter of the function.
 * @param args Argument values of the function parameters.
 * @param kwds Keywords of the function parameters.
 * @return A numpy array with the classifier output.
 */

static PyObject *predict (PyObject *dummy, PyObject *args, PyObject *kwds);

/**
 * @brief Python extension to save a trained model in the file system.
 *
 * This is the function of the python extension to save a model in the file system.
 *
 * @param dummy First parameter of the function.
 * @param args Argument values of the function parameters.
 * @param kwds Keywords of the function parameters.
 * @return An empty python object.
 */

static PyObject *save (PyObject *dummy, PyObject *args, PyObject *kwds);


/**
 * @brief Python extension to load a trained model from the file system.
 *
 * This is the function of the python extension to load a model from the file system.
 *
 * @param dummy First parameter of the function.
 * @param args Argument values of the function parameters.
 * @param kwds Keywords of the function parameters.
 * @return A python object that is a capsule of the C pointer to the trained model.
 */

static PyObject *load (PyObject *dummy, PyObject *args, PyObject *kwds);


/**
 * @brief Convert numpy array to a dataset.
 *
 * It converts a dataset in numpy array format to our internal format.
 *
 * @param arr1 Python numpy array with the training set features.
 * @param args Python numpy array with the training set labels.
 * @return The dataset in our format.
 */

svm_dataset numpy2dataset(PyObject *arr1,PyObject *arr2);

/**
 * @brief Convert numpy array to a dataset and the average of every class data
 *
 * It converts a dataset in numpy array format to our internal format and calculates the average of every class data.
 *
 * @param arr1 Python numpy array with the training set features.
 * @param args Python numpy array with the training set labels.
 * @return The dataset in our format.
 */

svm_dataset numpy2datasetWithAverage(PyObject *arr1,PyObject *arr2);

/**
 * @brief Python extension of the budgeted SVM training procedure using the parallel IRWLS algorithm.
 *
 * This is the function of the python extension to train a budgeted SVM using the parallel IRWLS algorithm.
 *
 * @param dummy First parameter of the function.
 * @param args Argument values of the function parameters.
 * @param kwds Keywords of the function parameters.
 * @return A python object that is a capsule of the C pointer to the trained model.
 */

static PyObject*
budgeted_train (PyObject *dummy, PyObject *args, PyObject *kwds);

/**
 * @brief Python extension of the SVM training procedure using the parallel IRWLS algorithm.
 *
 * This is the function of the python extension to train a SVM using the parallel IRWLS algorithm.
 *
 * @param dummy First parameter of the function.
 * @param args Argument values of the function parameters.
 * @param kwds Keywords of the function parameters.
 * @return A python object that is a capsule of the C pointer to the trained model.
 */

static PyObject*
full_train (PyObject *dummy, PyObject *args, PyObject *kwds);


/**
 * @brief List of methods.
 *
 * The list of python functions that can be called from the python module.
 */

static PyMethodDef LIBIRWLSMethods[] = {
    {"load", (PyCFunction)load, METH_VARARGS|METH_KEYWORDS, "It loads a model from a file"},
    {"full_train", (PyCFunction)full_train, METH_VARARGS|METH_KEYWORDS, "It trains a SVM using the parallel IRWLS procedure"},
    {"budgeted_train", (PyCFunction)budgeted_train, METH_VARARGS|METH_KEYWORDS, "It trains a budgeted SVM using the parallel IRWLS procedure"},
    {"save", (PyCFunction)save, METH_VARARGS|METH_KEYWORDS, "It saves a model in a file"},
    {"predict", (PyCFunction)predict, METH_VARARGS|METH_KEYWORDS, "Predictions using a trained model"},
    { NULL, NULL, 0, NULL}
};

/**
 * @brief Initialization of the python module.
 *
 * This function is called when the python extension is initialzed.
 *
 * @return The init function.
 */
PyMODINIT_FUNC
initLIBIRWLS(void);




