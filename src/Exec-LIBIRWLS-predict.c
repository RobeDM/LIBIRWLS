

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

    if(props.verbose==1){
        printf("\nRunning with parameters:\n");
        printf("------------------------\n");
        printf("Dataset: %s\n",data_file);
        printf("The model to use: %s\n",data_model);
        printf("The result will be saved in: %s\n",output_file);
        printf("flag l = %d (l = 1 for labeled datasets, l = 0 for unlabeled datasets)\n",props.Labels);
        printf("------------------------\n");
        printf("\n");
    }
  
    model  mymodel;
    
    // Reading the trained model from the file
    if(props.verbose==1) printf("\nReading trained model from file:%s\n",data_model);
    FILE *In = fopen(data_model, "rb");
    if (In == NULL) {
        fprintf(stderr, "Input file with the trained model not found: %s\n",data_model);
        exit(2);
    }
    readModel(&mymodel, In);
    fclose(In);
    if(props.verbose==1) printf("Model Loaded, it contains %d Support Vectors\n\n",mymodel.nSVs);


    // Loading dataset
    if(props.verbose==1) printf("Reading dataset from file:%s\n",data_file);

    In = fopen(data_file, "rb");
    if (In == NULL) {
        fprintf(stderr, "Input file with the training set not found: %s\n",data_file);
        exit(2);
    }
    fclose(In);	  

    svm_dataset dataset;
    if(props.Labels==0){
        if(props.file==1){
            dataset=readUnlabeledFile(data_file);
        }else{
            dataset=readUnlabeledFileCSV(data_file,props.separator);
        }
    }else{
        if(props.file==1){
            dataset=readTrainFile(data_file);			
        }else{
            dataset=readTrainFileCSV(data_file,props.separator);	
        }
    }
    if(props.verbose==1) printf("Dataset Loaded, it contains %d samples and %d features\n\n", dataset.l,dataset.maxdim);
    
    // Set the number of openmp threads
    omp_set_num_threads(props.Threads);

    //Making predictions
    if(props.verbose==1) printf("Classifying data...\n");
    double *predictions;
    if (props.Soft==0){
        predictions=test(dataset,mymodel,props);
    }else{
        predictions=softTest(dataset,mymodel,props);
    }
    if(props.verbose==1) printf("data classified\n");	
    if(props.verbose==1) printf("\nWriting output in file: %s \n\n",output_file);
    writeOutput (output_file, predictions,dataset.l);

    freeDataset(dataset);
    freeModel(mymodel);
    free(predictions);
    return 0;   
}
