

#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "kernels.h"
#include "LIBIRWLS-predict.h"


/**
 * @brief It the main function to build the executable file to make predictions 
 * on a dataset using a model previously trained using PIRWLS-train or PSIRWLS-train.
 */

int main(int argc, char** argv)
{
    // Parsing command line to extract parameters.
    predictProperties props = parsePredictParameters(&argc, &argv);
  
    //Show error msg if there are wrong parameters.
    if (argc != 4) {
        printPredictInstructions();
        return 4;
    }

    // The name of the files
    char * data_file = argv[1];
    char * data_model = argv[2];
    char * output_file = argv[3];

    printf("\nRunning with parameters:\n");
    printf("------------------------\n");
    printf("Dataset: %s\n",data_file);
    printf("The model to use: %s\n",data_model);
    printf("The result will be saved in: %s\n",output_file);
    printf("flag l = %d (l = 1 for labeled datasets, l = 0 for unlabeled datasets)\n",props.Labels);
    printf("------------------------\n");
    printf("\n");

  
    model  mymodel;

    
    // Reading the trained model from the file
    printf("\nReading trained model from file:%s\n",data_model);
    FILE *In = fopen(data_model, "rb");
    if (In == NULL) {
        fprintf(stderr, "Input file with the trained model not found: %s\n",data_model);
        exit(2);
    }
    readModel(&mymodel, In);
    fclose(In);
    printf("Model Loaded, it contains %d Support Vectors\n\n",mymodel.nSVs);


    // Loading dataset
    printf("Reading dataset from file:%s\n",data_file);
    svm_dataset dataset;
    In = fopen(data_file, "rb");
    if (In == NULL) {
        fprintf(stderr, "Input file with the training set not found: %s\n",data_file);
        exit(2);
    }
    fclose(In);	  
    if(props.Labels==0){
        dataset=readUnlabeledFile(data_file);
    }else{
        dataset=readTrainFile(data_file);			
    }
    printf("Dataset Loaded, it contains %d samples and %d features\n\n", dataset.l,dataset.maxdim);

    
    // Set the number of openmp threads
    omp_set_num_threads(props.Threads);

    //Making predictions
    printf("Classifying data...\n");
    double *predictions;
    if (props.Soft==0){
        predictions=test(dataset,mymodel,props);
    }else{
        predictions=softTest(dataset,mymodel,props);
    }
    printf("data classified\n");	
    printf("\nWriting output in file: %s \n\n",output_file);
    writeOutput (output_file, predictions,dataset.l);
    return 0;
    
}
