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
 * @brief Functions to train a full SVM using the IRWLS algorithm.
 *
 * For a detailed description of the algorithm and its parameters read the following paper: \n Pérez-Cruz, F., Alarcón-Diana, P. L., Navia-Vázquez, A., & Artés-Rodríguez, A. (2001). Fast Training of Support Vector Classifiers. In Advances in Neural Information Processing Systems (pp. 734-740)
 *
 * For a detailed description about the parallelization read the following paper: \n
 Díaz-Morales, R., & Navia-Vázquez, Á. (2016). Efficient parallel implementation of kernel methods. Neurocomputing, 191, 175-186.
 *
 * @file PIRWLS-train.h
 * @author Roberto Diaz Morales
 * @date 23 Aug 2016
 */


#ifndef PIRWLSTRAIN_
#define PIRWLSTRAIN_

#include "IOStructures.h"

/**
 * @brief Random permutation of n elements.
 *
 * It crates a random permutation of n elements.
 *
 * @param n The number of elementos in the permutation.
 * @return The permutation.
 */

int * rpermute(int n);

/**
 * @brief IRWLS procedure on a Working Set.
 *
 * For a detailed description of the algorithm and its parameters read the following paper: \n Pérez-Cruz, F., Alarcón-Diana, P. L., Navia-Vázquez, A., & Artés-Rodríguez, A. (2001). Fast Training of Support Vector Classifiers. In Advances in Neural Information Processing Systems (pp. 734-740)
 *
 * For a detailed description about the parallelization read the following paper: \n
 Díaz-Morales, R., & Navia-Vázquez, Á. (2016). Efficient parallel implementation of kernel methods. Neurocomputing, 191, 175-186.
 *
 * @param dataset The training dataset.
 * @param props The strut of training properties.
 * @param GIN The classification effect of the inactive set.
 * @param e The current error on every training data.
 * @param beta The bias term of the classification function.
 * @return The new weights vector of the classifier.
 */

double* subIRWLS(svm_dataset dataset,properties props, double *GIN, double *e, double *beta);

/**
 * @brief It trains a full SVM with a training set.
 *
 *  For a detailed description of the algorithm and its parameters read the following paper: \n Pérez-Cruz, F., Alarcón-Diana, P. L., Navia-Vázquez, A., & Artés-Rodríguez, A. (2001). Fast Training of Support Vector Classifiers. In Advances in Neural Information Processing Systems (pp. 734-740)
 *
 * For a detailed description about the parallelization read the following paper: \n
 Díaz-Morales, R., & Navia-Vázquez, Á. (2016). Efficient parallel implementation of kernel methods. Neurocomputing, 191, 175-186.
 *
 * It trains a full SVM using a training set and the training parameters.
 * @param dataset The training set.
 * @param props The values of the training parameters.
 * @return The weights of every Support Vector of the SVM.
 */

double* trainFULL(svm_dataset dataset,properties props);


/**
 * @brief Print Instructions.
 *
 *  It shows PIRWLS-train command line instructions in the standard output.
 */

void printPIRWLSInstructions(void) ;


/**
 * @brief It parses the command line.
 *
 * It parses input command line to extract the parameters.
 * @param argc The number of words of the command line.
 * @param argv The list of words of the command line.
 * @return A struct that contains the values of the training parameters.
 */

properties parseTrainPIRWLSParameters(int* argc, char*** argv);

/**
 * @brief It converts the result into a model struct.
 *
 * After the training of a SVM using the IRWLS procedure, this function build a struct with the information and returns it.
 *
 * @param props The training parameters.
 * @param dataset The training set.
 * @param beta The weights of the classifier.
 * @return The struct that storages all the information of the classifier.
 */

model calculatePIRWLSModel(properties props, svm_dataset dataset, double * beta );

#endif


