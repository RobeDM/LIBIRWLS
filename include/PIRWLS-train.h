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
 * @file PIRWLS-train.h
 * @author Roberto Diaz Morales
 * @date 23 Aug 2016
 * @brief Functions to train a full SVM using the IRWLS algorithm.
 *
 * Functions to train a full SVM using the IRWLS algorithm.
 */


#ifndef PIRWLSTRAIN_
#define PIRWLSTRAIN_

#include "IOStructures.h"

/**
 * @brief It crates a random permutation of n elements.
 *
 * It crates a random permutation of n elements.
 */

int * rpermute(int n);

/**
 * @brief It uses the IRWLS procedure on a Working Set.
 *
 * It uses the IRWLS procedure on a Working Set.
 */

double* subIRWLS(svm_dataset dataset,properties props, double *GIN, double *e, double *beta);


/**
 * @brief It trains a full SVM with a training set.
 *
 * It trains a full SVM with a training set.
 */

double* trainFULL(svm_dataset dataset,properties props);


/**
 * @brief It shows PIRWLS-train command line instructions in the standard output.
 *
 *  It shows PIRWLS-train command line instructions in the standard output.
 */

void printPIRWLSInstructions() ;


/**
 * @brief It parses input command line to extract the parameters.
 *
 * It parses input command line to extract the parameters.
 */

properties parseTrainPIRWLSParameters(int* argc, char*** argv);

/**
 * @brief It converts the result into a model struct.
 *
 * It converts the result into a model struct.
 */

model calculatePIRWLSModel(properties props, svm_dataset dataset, double * beta );

#endif


