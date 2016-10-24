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
 * @file LIBIRWLS-predict.h
 * @author Roberto Diaz Morales
 * @date 23 Aug 2016
 * @brief Functions to classify data with a trained model.
 *
 * Functions to classify data with a trained model.
 */


#ifndef LIBIRWLSPREDICT_
#define LIBIRWLSPREDICT_

#include "IOStructures.h"

/**
 * @brief Function to classify data in a labeled dataset and to obtain the accuracy.
 *
 * Function that uses a trained model on a dataset and obtains the class of every training sample.
 * @param dataset The test set.
 * @param mymodel A trained SVM model.
 * @param props The test properties.
 * @return The output of the classifier for every test sample (soft output).
 */

double *test(svm_dataset dataset, model mymodel,predictProperties props);


/**
 * @brief Function to obtain the soft output of the classifier.
 *
 * Function to obtain the soft output (the output of the classifier before using the threshold to decide class +1 or -1) of the model on a dataset. It is useful to combine this output with other algorithms in ensembles.
 * @param dataset The test set.
 * @param mymodel A trained SVM model.
 * @param props The test properties.
 * @return The output of the classifier for every test sample (soft output).
 */

double *softTest(svm_dataset dataset, model mymodel,predictProperties props);


/**
 * @brief Print instructions.
 *
 * It shows the command line instructions in the standard output.
 */

void printPredictInstructions(void);


/**
 * @brief It parses the prediction parameters from the command line.
 *
 * It reads the command line, extract the parameters and creates a strict with the value of its values.
 * @param argc The number of words of the command line.
 * @param argv The list of words of the command line.
 * @return A struct that contains the values of the test parameters.
 */

predictProperties parsePredictParameters(int* argc, char*** argv);


#endif

