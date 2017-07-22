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
 * @brief Implementation of the functions of the python extension.
 *
 * See  pythonextension.h for a detailed description of the functions and parameters.
 * 
 * @file pythonmodule.c
 * @author Roberto Diaz Morales
 * @date 23 Sep 2016
 * @see pythonmodule.h
 * 
 */

#include "Python.h"
#include "numpy/arrayobject.h"
#include "IOStructures.h"
#include "full-train.h"
#include "budgeted-train.h"
#include "ParallelAlgorithms.h"
#include "LIBIRWLS-predict.h"
#include <omp.h>
#include "pythonmodule.h"


/**
 * @brief Model capsule destructor
 *
 * The capsule destructor of the model object
 *
 * @param pymodel The model to delete
 */
void delModel(PyObject* pymodel) {
    model *modelo = (model*)PyCapsule_GetPointer(pymodel, "CLASSIFIER");
    freeModel(modelo[0]);
    free(modelo);
}


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

static PyObject *predict (PyObject *dummy, PyObject *args, PyObject *kwds)
{
    predictProperties props;
    props.Threads=1;
    props.Soft=0;
    props.verbose=1;
    svm_dataset dataset;

    static char *kwlist[] = {"classifier", "data", "labels", "threads", "Soft","verbose", NULL};

    PyObject *pymodel=NULL, *arg1=NULL, *arr1=NULL, *arg2=NULL, *arr2=NULL;
    props.Threads=1;
    props.Soft=0;
        
    if (!PyArg_ParseTupleAndKeywords(args,kwds, "OO|Oiii",kwlist, &pymodel,&arg1,&arg2,&props.Threads,&props.Soft,&props.verbose))
    return NULL;

    model *modelo;
    if (!(modelo = (model*) PyCapsule_GetPointer(pymodel, "CLASSIFIER"))) return NULL;
    
    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr1 == NULL) return NULL;

    npy_intp dims[1];
    npy_intp *shape = PyArray_DIMS(arr1);
    dims[0] = shape[0];

    if (arg2 == NULL){
        arr2 = PyArray_SimpleNewFromData(1,dims,NPY_DOUBLE,(double *) calloc(dims[0],sizeof(double)));
        props.Labels=0;
    }else{
        arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
        props.Labels=1;
        
    }
    
    dataset = numpy2dataset(arr1,arr2);

    omp_set_num_threads(props.Threads);

    double *predictions;
    if(props.Soft==0) predictions = test(dataset, modelo[0], props);
    else predictions = softTest(dataset, modelo[0], props);

    Py_DECREF(arr1);
    Py_DECREF(arr2);

    freeDataset(dataset);    

    PyObject *predictionsNP = PyArray_SimpleNewFromData(1,dims,NPY_DOUBLE,predictions);

    return predictionsNP;

}

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

static PyObject *save (PyObject *dummy, PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"classifier", "file", NULL};

    PyObject *pymodel=NULL;
    const char *filename;
        
    if (!PyArg_ParseTupleAndKeywords(args,kwds, "Os",kwlist, &pymodel,&filename))
    return NULL;

    model *modelo;
    if (!(modelo = (model*) PyCapsule_GetPointer(pymodel, "CLASSIFIER"))) return NULL;
    
    FILE *Out = fopen(filename, "wb");
    storeModel(modelo, Out);
    fclose(Out);
    
    Py_INCREF(Py_None);
    return Py_None;

}


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

static PyObject *load (PyObject *dummy, PyObject *args, PyObject *kwds)
{
    const char *filename;
     
    if (!PyArg_ParseTuple(args, "s", &filename))    
    return NULL;


    model  mymodel;

    FILE *In = fopen(filename, "rb");
    if (In == NULL) {
        fprintf(stderr, "Input file with the trained model not found: %s\n",filename);
        exit(2);
    }

    readModel(&mymodel, In);
    fclose(In);

    model *numero = (model *) malloc((1)*sizeof(model));
    if (NULL == numero)
        return NULL;

    *numero=mymodel;
    PyObject * ret = PyCapsule_New(numero, "CLASSIFIER", NULL);
    return ret;

}


/**
 * @brief Convert numpy array to a dataset.
 *
 * It converts a dataset in numpy array format to our internal format.
 *
 * @param arr1 Python numpy array with the training set features.
 * @param args Python numpy array with the training set labels.
 * @return The dataset in our format.
 */

svm_dataset numpy2dataset(PyObject *arr1,PyObject *arr2){

    int i,e,eindex=0;
    npy_intp *shape = PyArray_DIMS(arr1);

    //Dataset struct initialization
    svm_dataset dataset;
    dataset.l = shape[0];
    dataset.maxdim=shape[1];
    dataset.y = (double *) calloc(dataset.l,sizeof(double));
    dataset.quadratic_value = (double *) calloc(dataset.l,sizeof(double));
    dataset.x = (svm_sample **) calloc(dataset.l,sizeof(svm_sample *));

    dataset.sparse=0;
    int elements=dataset.l;
    double *aux;

    // Counting the number of non zero values 
    for(i=0;i<dataset.l;i++){

        for(e=0;e<dataset.maxdim;e++){
            aux = (double*)PyArray_GETPTR2(arr1, i, e);
            if (aux[0] != 0.0){
                elements++;
                dataset.sparse=1;
            }
        }

        aux = (double *)PyArray_GETPTR1(arr2, i);
        dataset.y[i] = aux[0];
    }

    // Iteration over the numpy array storing the dataset
    dataset.features = (svm_sample *) calloc(elements,sizeof(svm_sample));

    for(i=0;i<dataset.l;i++){
        dataset.x[i] = &dataset.features[eindex];
        for(e=0;e<dataset.maxdim;e++){
            aux = (double*)PyArray_GETPTR2(arr1, i, e);
            if (aux[0] != 0.0){
                dataset.quadratic_value[i] += pow(aux[0],2);
                dataset.features[eindex].index=(e+1);
                dataset.features[eindex].value=aux[0];
                eindex++;
            }
        }
        dataset.features[eindex].index=-1;
        eindex++;
    }
    
    return dataset;

}

/**
 * @brief Convert numpy array to a dataset and the average of every class data
 *
 * It converts a dataset in numpy array format to our internal format and calculates the average of every class data.
 *
 * @param arr1 Python numpy array with the training set features.
 * @param args Python numpy array with the training set labels.
 * @return The dataset in our format.
 */

svm_dataset numpy2datasetWithAverage(PyObject *arr1,PyObject *arr2){

    int i,e,eindex=0;

    npy_intp *shape = PyArray_DIMS(arr1);

    // Dataset Initialization
    svm_dataset dataset;
    dataset.l = shape[0];
    dataset.maxdim=shape[1];
    dataset.y = (double *) calloc(dataset.l+2,sizeof(double));
    dataset.quadratic_value = (double *) calloc(dataset.l+2,sizeof(double));
    dataset.x = (svm_sample **) calloc(dataset.l+2,sizeof(svm_sample *));

    dataset.sparse=0;
    int elements=dataset.l,positives=0,negatives=0;
    double *aux;

    // Counting the number of non zero values
    for(i=0;i<dataset.l;i++){

        for(e=0;e<dataset.maxdim;e++){
            aux = (double*)PyArray_GETPTR2(arr1, i, e);
            if (aux[0] != 0.0){
                elements++;
                dataset.sparse=1;
            }
        }

        aux = (double *)PyArray_GETPTR1(arr2, i);
        dataset.y[i] = aux[0];

        if(aux[0]==1.0) positives++;
        else negatives++;
        
    }

    dataset.features = (svm_sample *) calloc(elements+2*(dataset.maxdim+1),sizeof(svm_sample));

    dataset.y[dataset.l]=1.0;
    dataset.y[dataset.l+1]=-1.0;
    dataset.x[dataset.l]=&dataset.features[elements];
    dataset.x[dataset.l+1]=&dataset.features[elements+dataset.maxdim+1];

    for(e=0;e<dataset.maxdim;e++){
        dataset.features[elements+e].index=(e+1);
        dataset.features[elements+e].value=0.0;
        dataset.features[elements+dataset.maxdim+1+e].index=(e+1);
        dataset.features[elements+dataset.maxdim+1+e].value=0.0;
    }
    dataset.features[elements+dataset.maxdim+1].index=-1;
    dataset.features[elements+2*dataset.maxdim+1].index=-1;


    // Iteration over the numpy array storing the values in the dataset
    for(i=0;i<dataset.l;i++){
        dataset.x[i] = &dataset.features[eindex];
        for(e=0;e<dataset.maxdim;e++){
            aux = (double*)PyArray_GETPTR2(arr1, i, e);
            if (aux[0] != 0.0){
                dataset.quadratic_value[i] += pow(aux[0],2);
                dataset.features[eindex].index=(e+1);
                dataset.features[eindex].value=aux[0];
                if(dataset.y[i]==1.0) dataset.features[elements+e].value+=(aux[0]/positives);
                else dataset.features[elements+dataset.maxdim+1+e].value+=(aux[0]/negatives);
                eindex++;
            }
        }
        dataset.features[eindex].index=-1;
        eindex++;
    }

    for(e=0;e<dataset.maxdim;e++){
        dataset.quadratic_value[dataset.l] += dataset.features[elements+e].value;
        dataset.quadratic_value[dataset.l+1] += dataset.features[elements+dataset.maxdim+1+e].value;
    }    

    return dataset;

}

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
budgeted_train (PyObject *dummy, PyObject *args, PyObject *kwds)
{

    // Pointers to store the dataset
    PyObject *arg1=NULL, *arr1=NULL, *arg2=NULL, *arr2=NULL;

    // The properties struct used to train the algorithm
    properties props;
    props.Kgamma = 1.0;
    props.C = 1.0;
    props.Threads=1;
    props.MaxSize=0;
    props.Eta=0.001;
    props.size=10;
    props.algorithm=0;
    props.kernelType=1;
    props.verbose=1;
    
    // List of keywords parameters.
    static char *kwlist[] = {"data","labels","gamma", "C", "threads", "size", "algorithm", "kernel","verbose", NULL};

    // It parses the parameters
    if (!PyArg_ParseTupleAndKeywords(args,kwds, "OO|ddiiiii", kwlist, &arg1, &arg2, &props.Kgamma, &props.C, &props.Threads, &props.size, &props.algorithm, &props.kernelType,&props.verbose))
    return NULL;  

    // Obtaining the dataset
    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr1 == NULL)
    return NULL;

    arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr2 == NULL)
    return NULL;
    
    // Transforming the numpy dataset to our format.
    svm_dataset dataset = numpy2datasetWithAverage(arr1, arr2);

    // Obtaining the centroids
    omp_set_num_threads(props.Threads);
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
    initMemory(props.Threads,props.size);
    int * centroids;
    if (props.algorithm==0){
        centroids=randomCentroids(dataset,props);
    }else{
        centroids=SGMA(dataset,props);
    }

    // Using the IRWLS algorithm
    omp_set_num_threads(props.Threads);
    double * W = IRWLSpar(dataset,centroids,props);
    model modelo = calculateBudgetedModel(props, dataset,centroids, W);

    // Decref the created python objects
    Py_DECREF(arr1);
    Py_DECREF(arr2);

    // Creating the python object to save the trained model
    model *numero = (model *) malloc((1)*sizeof(model));
    if (NULL == numero)
        return NULL;

    freeDataset(dataset);

    *numero=modelo;
    PyObject * ret = PyCapsule_New(numero, "CLASSIFIER", NULL);
    return ret;
}

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
full_train (PyObject *dummy, PyObject *args, PyObject *kwds)
{

    //Pointers to store the dataset
    PyObject *arg1=NULL, *arr1=NULL, *arg2=NULL, *arr2=NULL;

    //The properties struct used to train the algorithm
    properties props;
    props.Kgamma = 1.0;
    props.C = 1.0;
    props.Threads=1;
    props.MaxSize=500;
    props.Eta=0.001;
    props.size=10;
    props.kernelType=1;
    props.verbose=1;

    //List of keyword parameters.
    static char *kwlist[] = {"data","labels","gamma", "C", "threads", "workingSet", "eta", "kernel","verbose", NULL};

    //It parses the parameters
    if (!PyArg_ParseTupleAndKeywords(args,kwds, "OO|ddiidii",kwlist,&arg1,&arg2,&props.Kgamma,&props.C,&props.Threads,&props.MaxSize,&props.Eta,&props.kernelType,&props.verbose))
    return NULL;  

    //Obtaining the numpy dataset
    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr1 == NULL)
    return NULL;

    arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr2 == NULL)
    return NULL;
    
    //Transforming the numpy dataset to our format.
    svm_dataset dataset = numpy2dataset(arr1, arr2);

    //Using the IRWLS algorithm
    initMemory(props.Threads,(props.MaxSize+1));  
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
    double * W = trainFULL(dataset,props);
    model modelo = calculateFULLModel(props, dataset, W);

    //Decref the created python objects
    Py_DECREF(arr1);
    Py_DECREF(arr2);

    //Creating the python object to save the trained model
    model *numero = (model *) malloc((1)*sizeof(model));
    if (NULL == numero)
        return NULL;

    freeDataset(dataset);

    *numero=modelo;
    PyObject * ret = PyCapsule_New(numero, "CLASSIFIER", delModel);
    return ret;
}

/**
 * @brief Initialization of the python module.
 *
 * This function is called when the python extension is initialzed.
 *
 * @return The init function.
 */
PyMODINIT_FUNC
initLIBIRWLS(void)
{
     (void) Py_InitModule("LIBIRWLS", LIBIRWLSMethods);
     import_array();
}



