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


#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include<sys/time.h>
#include <string.h>


#ifdef USE_MKL
#include "mkl_cblas.h"
#include "mkl_blas.h"
#include "mkl.h"
#else
#include "cblas.h"
#endif

#include "../include/kernels.h"




double *test(svm_dataset dataset, model mymodel,predictProperties props){

    int i,j;		
    double *predictions=(double *) malloc((dataset.l)*sizeof(double));

    #pragma omp parallel default(shared) private(i,j)
    {	
    #pragma omp for schedule(static)			
    for (i=0;i<dataset.l;i++){
        double pred=mymodel.bias;
        for (j=0;j<mymodel.nSVs;j++){
            pred+=(mymodel.weights[j])*kernelTest(dataset, i,  mymodel, j);
        }
        predictions[i]=pred;
    }	
    }

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


void printPredictInstructions() {
    fprintf(stderr, "LIBIRWLS-predict: This software predicts the label of a SVM given a data set of samples and a model obtained with PIRWLS-train or PSIRWLS-train");
    fprintf(stderr, "and store the results in an output file.\n\n");
    fprintf(stderr, "Usage: PSIRWLS-predict [options] data_set_file model_file output_file\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t Number of Threads: (default 1)\n");
    fprintf(stderr, "  -l type of data set: (default 0)\n");
    fprintf(stderr, "       0 -- Data set with no target as first dimension.\n");
    fprintf(stderr, "       1 -- Data set with label as first dimension (obtains accuracy too)\n");
    fprintf(stderr, "Note:\n");
    fprintf(stderr, "       The data set file must have the same format as the data set\n");
    fprintf(stderr, "       given to PIRWLS-train.\n");
}


predictProperties parsePredictParameters(int* argc, char*** argv, int semiparametric) {

    predictProperties props;
    props.Labels=0;
    props.Threads=1;
	
    int i;
    for (i = 1; i < *argc; ++i) {
        if ((*argv)[i][0] != '-') break;
        if (++i >= *argc) {
            if (semiparametric==0)
                printPredictInstructions();
            else
                printPredictInstructions();
            exit(1);
        }

        char* param_name = &(*argv)[i-1][1];
        char* param_value = (*argv)[i];
        
        if (strcmp(param_name, "t") == 0) {    	
            props.Threads = atof(param_value);
        } else if (strcmp(param_name, "l") == 0) {
            props.Labels = atoi(param_value);
            if(props.Labels !=0 && props.Labels !=1){
      	        printf("\nInvalid type of test data set:%d\n",props.Labels);
                exit(2);
            }
        } else {
            fprintf(stderr, "Unknown parameter %s\n",param_name);
            if (semiparametric==0)
                printPredictInstructions();
            else
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


int main(int argc, char** argv)
{

    predictProperties props = parsePredictParameters(&argc, &argv,0);
  
    if (argc != 4) {
        printPredictInstructions();
        return 4;
    }

    char * data_file = argv[1];
    char * data_model = argv[2];
    char * output_file = argv[3];
  
    model  mymodel;
    
    //////////////////////////////
    FILE *In = fopen(data_model, "r+");
    readModel(&mymodel, In);
    fclose(In);

    printf("\nModel Loaded from file: %s\nSupport Vectors: %d\n\n",data_model,mymodel.nSVs);

    svm_dataset dataset;
	  
    if(props.Labels==0){
        dataset=readUnlabeledFile(data_file);
    }else{
        dataset=readTrainFile(data_file);			
    }

    printf("Dataset Loaded from file: %s\nTraining samples: %d\nNumber of features: %d\n\n",data_file, dataset.l,dataset.maxdim);

    omp_set_num_threads(props.Threads);
	
    double *predictions=test(dataset,mymodel,props);
	
    printf("\nWriting output in file: %s \n\n",output_file);
    writeOutput (output_file, predictions,dataset.l);
    return 0;

}
