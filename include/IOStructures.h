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
 * @file IOStructures.h
 * @author Roberto Diaz Morales
 * @date 23 Aug 2016
 * @brief Input and Output structures and procedures.
 *
 * Input and Output structures and procedures.
 */


#ifndef IOSTRUCTURES_
#define IOSTRUCTURES_

#include <stdio.h>

/**
 * @brief Training parameters of the IRWLS procedures.
 *
 * This struct stores the training parameters of the IRWLS procedures.
 */

typedef struct properties{
    double Kgamma; /**< Gamma parameter of the kernel function. */
    double kernelType; /**< The kernel function (linear=0, rbf=1). */
    double C; /**< C parameter of the SVM cost function. */
    int Threads; /**< Number of threads to parallelize the operations. */
    int MaxSize; /**< Maximum size of the active set to calculate the SVM. */
    int size; /**< Size of semiparametric model (if we are executing the semiparametric version). */
    double Eta; /**< Convergence criteria of the SVM. */
}properties;


/**
 * @brief Testing parameters of the IRWLS procedures.
 *
 * This struct stores the testing parameters of the IRWLS procedures.
 */

typedef struct predictProperties{
    int Labels; /**< If the dataset to test is labeled. */
    int Threads; /**< Number of threads to make the predictions on the dataset. */
    int Soft; /**< The classifier obtains Soft output or not. */
}predictProperties;


/**
 * @brief It represents a trained model that has been obtained using PIRWLS or PSIRWLS.
 *
 * This structures saves all the variables of a trained model needed to classify future data.
 */

typedef struct model{
    double Kgamma; /**< Gamma parameter of the kernel function. */
    int kernelType; /**< The kernel function (linear=0, rbf=1). */
    int sparse; /**< To tell if the datasets are sparse or not. */
    int nSVs; /**< To tell if the datasets are sparse or not. */
    int nElem; /**< Number of features distinct than zero in the dataset. */   
    double *weights; /**< The weight associated to every support vector. */   
    struct svm_sample **x; /**< The support vectors. */
    double *quadratic_value; /**< Array that contains the norm L2 of every support vector. */    
    int maxdim; /**< Number of dimensions of the dataset. */
    double bias; /**< The bias term of the classification function. */
}model;


/**
 * @brief A single feature of a data.
 *
 * This structure represents a single feature of a data. It is composed of a features index and its value.
 */

typedef struct svm_sample{
    int index; /**< The feature index. */   
    double value; /**< The feature value. */   
}svm_sample;


/**
 * @brief A dataset.
 *
 * This structure represents a dataset, a collection of samples and its associated labels.
 */

typedef struct svm_dataset{
    int l; /**< If the dataset is labeled or not. */   
    int sparse; /**< If the dataset is sparse or not. */   
    int maxdim; /**< The number of features of the dataset. */   
    double *y; /**< The label of every sample. */   
    struct svm_sample **x; /**< The samples. */   
    double *quadratic_value; /**< The L2 norm of every sample. It is used to compute kernel functions faster.*/   
}svm_dataset;


/**
 * @brief A comparator of two data.
 *
 * A function to compare two data and returns:
 *  * 1 if teh first argument is higher than the second one.
 *  * -1 if teh first argument is lower than the second one.
 *  * 0 if both arguments are equal.
 * @param a The first argument to compare.
 * @param b The second argument to compare.
 * @return The result of comparing a and b.
 */

static int compare (const void * a, const void * b);


/**
 * @brief It reads a file that contains a labeled dataset in libsvm format.
 *
 * It reads a file that contains a labeled dataset in libsvm format, the format is the following one:
 * +1 1:5 7:2 15:6
 * +1 1:5 7:2 15:6 23:1
 * -1 2:4 3:2 10:6 11:4
 * ...
 *
 * @param filename A string with the name of the file that contains the dataset.
 * @return The struct with the dataset information.
 */

svm_dataset readTrainFile(char filename[]);


/**
 * @brief It reads a file that contains an unlabeled dataset in libsvm format.
 *
 * It reads a file that contains an unlabeled dataset in libsvm format. The format si the following one:
 * 1:5 7:2 15:6
 * 1:5 7:2 15:6 23:1
 * 2:4 3:2 10:6 11:4
 * ...
 *
 * @param filename A string with the name of the file that contains the dataset.
 * @return The struct with the dataset information.
 */

svm_dataset readUnlabeledFile(char filename[]);


/**
 * @brief It stores a trained model into a file.
 *
 * It stores the struct of a trained model (that has been obtained using PIRWLS or PSIRWLS) into a file.
 * @param mod The struct with the model to store.
 * @param Output The name of the file.
 */

void storeModel(model * mod, FILE *Output);

/**
 * @brief It loads a trained model from a file.
 *
 * It loads a trained model (that has been obtained using PIRWLS or PSIRWLS) from a file.
 * @param mod The pointer with the struct to load results.
 * @param Input The name of the file.
 */

void readModel(model * mod, FILE *Input);


/**
 * @brief It writes the content of a double array into a file.
 *
 * It writes the content of a double array into a file. It is used to save the predictions of a model on a dataset.
 * @param fileoutput The name of the file.
 * @param predictions The array with the information to save.
 * @param size The length of the array.
 */

void writeOutput (char fileoutput[], double *predictions, int size);

#endif

