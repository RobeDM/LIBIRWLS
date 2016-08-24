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

#include "../include/IOStructures.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>


static int compare (const void * a, const void * b){
  if (*(double*)a < *(double*)b) return -1;
  else if (*(double*)a > *(double*)b) return +1;
  else return 0;  
}

svm_dataset readTrainFile(char filename[]){

    svm_dataset dataset;
	
    int arraysize=256;

    char *endptr;
    char *idx, *val, *label;

  	
    if (filename == NULL){
        fprintf(stderr, "File not specified");
        exit(2);
    }
		
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "File not found: %s\n",filename);
        exit(2);
    }	

    char fileline[100000];

    dataset.l = 0;
    int elements = 0;
    dataset.sparse = 0;

    int maxindexDS = 0;
    int index;

    while (fgets(fileline, 100000, file) != NULL){

        char *p = strtok(fileline," \t");


        while(1){
            idx = strtok(NULL,":");
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') break;
            else{    
                index = (int) strtol(idx,&endptr,10);
                if(index>maxindexDS) maxindexDS=index;
            }
            ++elements;
        }
        ++elements;
        ++dataset.l;
    }



    elements=elements+2*(maxindexDS+2);
    double *meanPositives = (double *) calloc(maxindexDS+1,sizeof(double));
    double *meanNegatives = (double *) calloc(maxindexDS+1,sizeof(double));
    double sumPositives=0.0;
    double sumNegatives=0.0;

    rewind(file);
    
    dataset.y = (double *) calloc(dataset.l+2,sizeof(double));
    dataset.quadratic_value = (double *) calloc(dataset.l+2,sizeof(double));
    dataset.x = (svm_sample **) calloc(dataset.l+2,sizeof(svm_sample *));
    svm_sample* features = (svm_sample *) calloc(elements,sizeof(svm_sample));
    dataset.maxdim=0;

    int max_index = 0;
    int i,j,dm=0;
    int inst_max_index;
    int errno;


    for(i=0;i<dataset.l;i++){

        inst_max_index = -1;
        if (fgets(fileline, 100000, file)== NULL){
            fprintf(stderr, "Error reading data file\n");
            exit(2);
        }

        dataset.x[i] = &features[j];
	    label = strtok(fileline," \t\n");

        if(label == NULL){
            fprintf(stderr, "Wrong file format\n");
            exit(2);
        }

        dataset.y[i] = strtod(label,&endptr);

        if (dataset.y[i]==1.0){
            sumPositives=sumPositives+1;
        }else{
            sumNegatives=sumNegatives+1;
        }

        if(endptr == label || *endptr != '\0'){
            fprintf(stderr, "Wrong file format\n");
            exit(2);
        }
        dm = 0;
        while(1){
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL) break;

            errno = 0;
            features[j].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || features[j].index <= inst_max_index){
                fprintf(stderr, "Wrong file format\n");
                exit(2);
            }else{
                inst_max_index = features[j].index;
            }
            if(features[dm].index != features[j].index){
                dataset.sparse=1;
            }

            errno = 0;
            features[j].value = strtod(val,&endptr);

            if (dataset.y[i]==1.0){
                meanPositives[features[j].index] += features[j].value;
            }else{
                meanNegatives[features[j].index] += features[j].value;
            }

            dataset.quadratic_value[i] += pow(strtod(val,&endptr),2);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr))){
                fprintf(stderr, "Wrong file format\n");
                exit(2);
            }
            ++dm;
            ++j;
        }

        if(inst_max_index > max_index){
            max_index = inst_max_index;
        }

        features[j++].index = -1;


    }

    dataset.y[dataset.l]=1.0;
    dataset.x[dataset.l] = &features[j];
    for (i=0;i<=maxindexDS;i++){
        if (meanPositives[i] != 0.0){
            features[j].index = i;
            features[j].value = meanPositives[i]/sumPositives;
            dataset.quadratic_value[dataset.l] += pow(meanPositives[i]/sumPositives,2);
            ++j;
        }
    }
    
    features[j].index = -1;
    ++j;


    dataset.y[dataset.l+1]=-1.0;
    dataset.x[dataset.l+1] = &features[j];
    for (i=0;i<=maxindexDS;i++){
        if (meanNegatives[i] != 0.0){
            features[j].index = i;
            features[j].value = meanNegatives[i]/sumNegatives;
            dataset.quadratic_value[dataset.l+1] += pow(meanNegatives[i]/sumPositives,2);
            ++j;
        }
    }
    
    features[j].index = -1;
    ++j;

    dataset.maxdim=max_index;
    fclose(file);
    return dataset;

}

svm_dataset readUnlabeledFile(char filename[]){

    svm_dataset dataset;
	
    int arraysize=256;
  	
    if (filename == NULL){
        fprintf(stderr, "File not specified");
        exit(2);
    }
		
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "File not found: %s\n",filename);
        exit(2);
    }	

    char fileline[100000];

    dataset.l = 0;
    int elements = 0;
    dataset.sparse = 0;

    while (fgets(fileline, 100000, file) != NULL){
        char *p = strtok(fileline," \t");
        ++elements;

        while(1){
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') break;
            ++elements;
        }
        ++elements;
        
        ++dataset.l;
    }

    rewind(file);
    
    dataset.y = (double *) calloc(dataset.l,sizeof(double));
    dataset.quadratic_value = (double *) calloc(dataset.l,sizeof(double));
    dataset.x = (svm_sample **) calloc(dataset.l,sizeof(svm_sample *));
    svm_sample* features = (svm_sample *) calloc(elements,sizeof(svm_sample));
    dataset.maxdim=0;

    int max_index = 0;
    int i,j,dm=0;
    char *endptr;
    char *idx, *val, *label;
    int inst_max_index;
    int errno;

    for(i=0;i<dataset.l;i++){

        inst_max_index = -1;
        if (fgets(fileline, 100000, file)== NULL){
            fprintf(stderr, "Error reading data file\n");
            exit(2);
        }

        dataset.x[i] = &features[j];

        dataset.y[i] = 0;

        dm = 0;

        idx = strtok(fileline,":");
        val = strtok(NULL," \t");

        while(1){

            if(val == NULL) break;

            features[j].index = (int) strtol(idx,&endptr,10);

            if(endptr == idx || *endptr != '\0' || features[j].index <= inst_max_index){
                fprintf(stderr, "Wrong file format\n");
                exit(2);
            }else{
                inst_max_index = features[j].index;
            }

            if(features[dm].index != features[j].index){
                dataset.sparse=1;
            }

            features[j].value = strtod(val,&endptr);
            dataset.quadratic_value[i] += pow(strtod(val,&endptr),2);

            if(endptr == val ||  (*endptr != '\0' && !isspace(*endptr))){
                fprintf(stderr, "Wrong file format\n");
                exit(2);
            }

            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            ++dm;
            ++j;
        }

        if(inst_max_index > max_index){
            max_index = inst_max_index;
        }

        features[j++].index = -1;

    }

    dataset.maxdim=max_index;
    fclose(file);
    return dataset;

}


void storeModel(model * mod, FILE *Output){

    int aux;
    aux=fwrite(&mod->Kgamma, sizeof(double), 1, Output);    
    aux=fwrite(&mod->bias, sizeof(double), 1, Output);
    aux=fwrite(&mod->maxdim, sizeof(int), 1, Output);
    aux=fwrite(&mod->sparse, sizeof(int), 1, Output);
    aux=fwrite(&mod->nSVs, sizeof(int), 1, Output);
    aux=fwrite(&mod->nElem, sizeof(int), 1, Output);
    aux=fwrite(mod->weights, (mod->nSVs)*sizeof(double), 1, Output);
    aux=fwrite(mod->quadratic_value, (mod->nSVs)*sizeof(double), 1, Output);
    aux=fwrite(mod->x[0], (mod->nElem)*sizeof(svm_sample), 1, Output);

}

void readModel(model * mod, FILE *Input){
	
    int aux;
    aux=fread(&mod->Kgamma, sizeof(double), 1, Input);
    aux=fread(&mod->bias, sizeof(double), 1, Input);
    aux=fread(&mod->maxdim, sizeof(int), 1, Input);
    aux=fread(&mod->sparse, sizeof(int), 1, Input);
    aux=fread(&mod->nSVs, sizeof(int), 1, Input);    
    aux=fread(&mod->nElem, sizeof(int), 1, Input);    
    mod->weights = (double *)malloc((mod->nSVs)*sizeof(double));
    mod->quadratic_value = (double *)malloc((mod->nSVs)*sizeof(double));    
    aux=fread(mod->weights, (mod->nSVs)*sizeof(double), 1, Input);
    aux=fread(mod->quadratic_value, (mod->nSVs)*sizeof(double), 1, Input);
    mod->x = (svm_sample **)malloc((mod->nSVs)*sizeof(svm_sample *));    
    svm_sample* features = (svm_sample *) calloc((mod->nElem),sizeof(svm_sample));    
    aux=fread(features, (mod->nElem)*sizeof(svm_sample), 1, Input);

    mod->x[0]=&features[0];    
    int iterSV=1;
    for(aux=0;aux<(mod->nElem);aux++){
        if (features[aux].index == -1){
            if(iterSV<mod->nSVs) mod->x[iterSV]=&features[aux+1];
            ++iterSV;
        }
    }
}

void writeOutput (char fileoutput[], double *predictions, int size){
	
     FILE *Archivo;
     Archivo = fopen(fileoutput,"w+"); 
         
     if(Archivo !=0){
          int i;
          for(i=0;i<size;i++){
      	      fprintf(Archivo,"%lf\n",predictions[i]);
          }
     }
     fclose(Archivo);
}



