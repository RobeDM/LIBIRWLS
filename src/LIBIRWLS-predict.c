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
 * @brief Implementation of the predictions functions to classify data using a trained model.
 *
 * See LIBIRWLS-predict.h for a detailed description of its functions and parameters.
 * @file LIBIRWLS-predict.c
 * @author Roberto Diaz Morales
 * @date 23 Aug 2016
 *
 * @see LIBIRWLS-predict.h
 */

#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "kernels.h"
#include "LIBIRWLS-predict.h"

/**
 * @cond
 */

/**
 * @brief Function to obtain the soft output of the classifier.
 *
 * Function to obtain the soft output (the output of the classifier before using the threshold to decide +1 or -1) of the model on a dataset.
 * @param dataset The test set.
 * @param mymodel A trained SVM model.
 * @param props The test properties.
 * @return The output of the classifier for every test sample (soft output).
 */

double *softTest(svm_dataset dataset, model mymodel,predictProperties props){

    int i,j;		
    double *predictions=(double *) malloc((dataset.l)*sizeof(double));

    #pragma omp parallel default(shared) private(i,j)
    {	
    #pragma omp for schedule(static)
    for (i=0;i<dataset.l;i++){
        // Iteration over all the training elements
        double pred=mymodel.bias;
        for (j=0;j<mymodel.nSVs;j++){
            // Iteration over the Support Vectors
            pred+=(mymodel.weights[j])*kernelTest(dataset, i,  mymodel, j);
        }
        predictions[i]=pred;
    }	
    }

    // Obtaining accuracy (only for labeled test dataset)
    double aciertos=0.0;
    double total=(double)dataset.l;
    if(props.Labels==1){
        for (i=0;i<dataset.l;i++){
            if(predictions[i]>0 & dataset.y[i]>0) aciertos++;
            if(predictions[i]<=0 & dataset.y[i]<=0) aciertos++;
        }
        printf("Accuracy: %f\n",aciertos/total);
    }		
    return predictions;
}


/**
 * @brief Function to classify data in a labeled dataset and to obtain the accuracy.
 *
 * Function to classify data in a labeled dataset and to obtain the accuracy.
 * @param dataset The test set.
 * @param mymodel A trained SVM model.
 * @param props The test properties.
 * @return The output of the classifier for every test sample.
 */

double *test(svm_dataset dataset, model mymodel,predictProperties props){

    int i,j;		
    double *predictions=(double *) malloc((dataset.l)*sizeof(double));

    #pragma omp parallel default(shared) private(i,j)
    {	
    #pragma omp for schedule(static)
    for (i=0;i<dataset.l;i++){
        // Iteration over all the training elements
        double pred=mymodel.bias;
        for (j=0;j<mymodel.nSVs;j++){
            // Iteration over the Support Vectors
            pred+=(mymodel.weights[j])*kernelTest(dataset, i,  mymodel, j);
        }
        predictions[i]=pred;
        if(predictions[i]>=0.0) predictions[i]=1.0;
        else predictions[i]=-1.0;
    }	
    }

    // Obtaining accuracy (only for labeled test dataset)
    double aciertos=0.0;
    double total=(double)dataset.l;
    if(props.Labels==1){
        for (i=0;i<dataset.l;i++){
            if(predictions[i]>0 & dataset.y[i]>0) aciertos++;
            if(predictions[i]<=0 & dataset.y[i]<=0) aciertos++;
        }
        printf("Accuracy: %f\n",aciertos/total);
    }		
    return predictions;
}


/**
 * @brief It shows the command line instructions in the standard output.
 *
 * It shows the command line instructions in the standard output.
 */

void printPredictInstructions(void) {
    fprintf(stderr, "LIBIRWLS-predict: This software predicts the label of a SVM given a data set of samples and a model obtained with PIRWLS-train or PSIRWLS-train");
    fprintf(stderr, "and store the results in an output file.\n\n");
    fprintf(stderr, "Usage: PSIRWLS-predict [options] data_set_file model_file output_file\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t Number of Threads: (default 1)\n");
    fprintf(stderr, "  -l type of data set: (default 0)\n");
    fprintf(stderr, "       0 -- Data set with no target as first dimension.\n");
    fprintf(stderr, "       1 -- Data set with label as first dimension (obtains accuracy too)\n");
    fprintf(stderr, "  -s Soft output: (default 0)\n");
    fprintf(stderr, "       0 -- Obtains the class of every data (It takes values of +1 or -1).\n");
    fprintf(stderr, "       1 -- The output before the class decision (Useful to combine in ensembles with other algorithms).\n");
    fprintf(stderr, "Note:\n");
    fprintf(stderr, "  -f file format: (default 1)\n"); 
    fprintf(stderr, "       0 -- CSV format (comma separator)\n");
    fprintf(stderr, "       1 -- libsvm format\n");   
    fprintf(stderr, "  -p separator: csv separator character (default \",\" if csv format is selected)\n");    
    fprintf(stderr, "  -v verbose: (default 1)\n");        
    fprintf(stderr, "       0 -- No screen messages\n");
    fprintf(stderr, "       1 -- Screen messages\n");
}

/**
 * @brief It parses the prediction parameters from the command line.
 *
 * It parses the prediction parameters from the command line.
 * @param argc The number of words of the command line.
 * @param argv The list of words of the command line.
 * @return A struct that contains the values of the test parameters.
 */

predictProperties parsePredictParameters(int* argc, char*** argv) {

    predictProperties props;
    props.Labels=0;
    props.Threads=1;
    props.Soft=0;
    props.file = 1;
    props.separator = ",";
    props.verbose = 1;
	
    int i;
    for (i = 1; i < *argc; ++i) {
        if ((*argv)[i][0] != '-') break;
        if (++i >= *argc) {
            printPredictInstructions();
            exit(1);
        }

        char* param_name = &(*argv)[i-1][1];
        char* param_value = (*argv)[i];
        
        if (strcmp(param_name, "t") == 0) {    	
            props.Threads = atof(param_value);
        } else if (strcmp(param_name, "f") == 0) {
            props.file = atoi(param_value);
        } else if (strcmp(param_name, "p") == 0) {
            props.separator = param_value;
        } else if (strcmp(param_name, "v") == 0) {
            props.verbose = atoi(param_value);
        } else if (strcmp(param_name, "l") == 0) {
            props.Labels = atoi(param_value);
            if(props.Labels !=0 && props.Labels !=1){
      	        printf("\nInvalid type of test data set:%d\n",props.Labels);
                exit(2);
            }
        } else if (strcmp(param_name, "s") == 0) {
            props.Soft = atoi(param_value);
            if(props.Soft !=0 && props.Soft !=1){
      	        printf("\nInvalid type of output (-s param):%d\n",props.Soft);
                exit(2);
            }
        } else {
            fprintf(stderr, "Unknown parameter %s\n",param_name);
            printPredictInstructions();
            exit(2);
        }
    }
	  int j;
    for (j = 1; i + j - 1 < *argc; ++j) {
        (*argv)[j] = (*argv)[i + j - 1];
    }
    *argc -= i - 1;
    
    return props;
}


/**
 * @endcond
 */
