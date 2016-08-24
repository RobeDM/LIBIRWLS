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
 * @brief Structure to save training parameters.
 *
 * Structure to save training parameters.
 */

typedef struct properties{
    double Kgamma;
    double C;
    int Threads;
    int MaxSize;
    int size;
    double Eta;
}properties;


/**
 * @brief Structure to save predictions parameters.
 *
 * Structure to save prediction parameters.
 */

typedef struct predictProperties{
    int Labels;
    int Threads;
}predictProperties;


/**
 * @brief Structure of a trained model.
 *
 * Structure of a trained model.
 */

typedef struct model{
    double Kgamma;
    int sparse;
    int nSVs;
    int nElem;
    double *weights;    
    struct svm_sample **x;
    double *quadratic_value;    
    int maxdim;
    double bias;
}model;


/**
 * @brief This structure represents a single feature of a data.
 *
 * This structure represents a single feature of a data.
 */

typedef struct svm_sample{
    int index;
    double value;
}svm_sample;


/**
 * @brief This structure represents a dataset.
 *
 * This structure represents a dataset.
 */

typedef struct svm_dataset{
    int l;
    int sparse;
    int maxdim;
    double *y;
    struct svm_sample **x;
    double *quadratic_value;
}svm_dataset;


/**
 * @brief A comparator of two data.
 *
 * A function to compare two data.
 */

static int compare (const void * a, const void * b);


/**
 * @brief It reads a file that contains a labeled dataset in libsvm format.
 *
 * It reads a file that contains a labeled dataset in libsvm format.
 */

svm_dataset readTrainFile(char filename[]);


/**
 * @brief It reads a file that contains an unlabeled dataset in libsvm format.
 *
 * It reads a file that contains an unlabeled dataset in libsvm format.
 */

svm_dataset readUnlabeledFile(char filename[]);


/**
 * @brief It stores a trained model into a file.
 *
 * It stores a trained model into a file.
 */

void storeModel(model * mod, FILE *Output);

/**
 * @brief It loads a trained model from a file.
 *
 * It loads a trained model from a file.
 */

void readModel(model * mod, FILE *Input);


/**
 * @brief It writes the content of a double array into a file.
 *
 * It writes the content of a double array into a file.
 */

void writeOutput (char fileoutput[], double *predictions, int size);

#endif

